# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

# Adapted from Eventual-Inc/Daft's document embedding benchmark.
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import daft
import pymupdf
import ray
import torch
from daft import col
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DATA_ROOT = Path("/data/multimodal_inference_benchmarks/digitalcorpora")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "metadata")).expanduser()
PDF_ROOT = Path(os.environ.get("LOCAL_PDF_ROOT", DATA_ROOT / "pdf_dump")).expanduser()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", f"/tmp/daft_document_{uuid.uuid4().hex}")).expanduser()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))
INPUT_LIMIT = int(os.environ.get("INPUT_LIMIT", "0"))

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_PDF_PAGES = 100
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


def _parquet_input(path: Path) -> str:
    return str(path / "**/*.parquet") if path.is_dir() else str(path)


def _local_pdf_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return str(path)
    normalized = value.replace("\\", "/")
    marker = "pdf_dump/"
    relative = normalized.split(marker, 1)[1] if marker in normalized else path.name
    return str(PDF_ROOT / relative)


def _model_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, local_files_only=True)


@ray.remote
def warmup():
    pass


def extract_text_from_pdf(pdf_bytes):
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if len(document) > MAX_PDF_PAGES:
                return None
            return [{"text": page.get_text(), "page_number": int(page.number)} for page in document]
    except Exception as exc:
        print(f"Error extracting text from PDF: {exc}", file=sys.stderr, flush=True)
        return None


def chunk(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return [{"text": value, "chunk_id": chunk_id} for chunk_id, value in enumerate(splitter.split_text(text))]


@daft.udf(
    return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), EMBEDDING_DIM),
    concurrency=NUM_GPU_NODES,
    num_gpus=1.0,
    batch_size=BATCH_SIZE,
)
class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(_model_path(), device=device)
        self.model.compile()

    def __call__(self, text_col):
        if len(text_col) == 0:
            return []
        embeddings = self.model.encode(
            text_col.to_pylist(),
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        return embeddings.cpu().numpy()


def main() -> None:
    daft.context.set_runner_ray()
    ray.get([warmup.remote() for _ in range(64)])

    start = time.time()
    df = daft.read_parquet(_parquet_input(INPUT_PATH))
    if INPUT_LIMIT > 0:
        df = df.limit(INPUT_LIMIT)
    df = df.where(col("file_name").str.endswith(".pdf"))
    df = df.with_column(
        "uploaded_pdf_path",
        df["uploaded_pdf_path"].apply(_local_pdf_path, return_dtype=daft.DataType.string()),
    )
    df = df.with_column("pdf_bytes", df["uploaded_pdf_path"].url.download())

    page_type = daft.DataType.struct(fields={"text": daft.DataType.string(), "page_number": daft.DataType.int32()})
    df = df.with_column(
        "pages",
        df["pdf_bytes"].apply(
            extract_text_from_pdf,
            return_dtype=daft.DataType.list(page_type),
        ),
    )
    df = df.explode("pages")
    df = df.with_columns({"page_text": col("pages")["text"], "page_number": col("pages")["page_number"]})
    df = df.where(col("page_text").not_null())

    chunk_type = daft.DataType.struct(fields={"text": daft.DataType.string(), "chunk_id": daft.DataType.int32()})
    df = df.with_column(
        "chunks",
        df["page_text"].apply(chunk, return_dtype=daft.DataType.list(chunk_type)),
    )
    df = df.explode("chunks")
    df = df.with_columns({"chunk": col("chunks")["text"], "chunk_id": col("chunks")["chunk_id"]})
    df = df.where(col("chunk").not_null())
    df = df.with_column("embedding", Embedder(df["chunk"]))
    df = df.select("uploaded_pdf_path", "page_number", "chunk_id", "chunk", "embedding")
    df.write_parquet(str(OUTPUT_PATH))

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
