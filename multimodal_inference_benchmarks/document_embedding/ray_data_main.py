# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import pymupdf
import ray
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DATA_ROOT = Path("/data/multimodal_inference_benchmarks/digitalcorpora")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "metadata")).expanduser()
PDF_ROOT = Path(os.environ.get("LOCAL_PDF_ROOT", DATA_ROOT / "pdf_dump")).expanduser()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", f"/tmp/ray_data_document_{uuid.uuid4().hex}")).expanduser()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))
INPUT_LIMIT = int(os.environ.get("INPUT_LIMIT", "0"))

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_PDF_PAGES = 100
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


def _local_pdf_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    normalized = value.replace("\\", "/")
    marker = "pdf_dump/"
    relative = normalized.split(marker, 1)[1] if marker in normalized else path.name
    return PDF_ROOT / relative


def _model_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, local_files_only=True)


@ray.remote
def warmup():
    pass


def to_local_pdf_path(row):
    row["uploaded_pdf_path"] = str(_local_pdf_path(row["uploaded_pdf_path"]))
    return row


def extract_text_from_pdf(row):
    path = row["uploaded_pdf_path"]
    try:
        with pymupdf.open(path) as document:
            if len(document) > MAX_PDF_PAGES:
                return
            for page in document:
                yield {
                    "uploaded_pdf_path": path,
                    "page_number": int(page.number),
                    "page_text": page.get_text(),
                }
    except Exception as exc:
        print(f"Error extracting text from PDF {path}: {exc}", file=sys.stderr, flush=True)


def chunker(row):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    for chunk_id, text in enumerate(splitter.split_text(row["page_text"])):
        yield {
            "uploaded_pdf_path": row["uploaded_pdf_path"],
            "page_number": row["page_number"],
            "chunk_id": chunk_id,
            "chunk": text,
        }


class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(_model_path(), device=device)
        self.model.compile()

    def __call__(self, batch):
        batch["embedding"] = self.model.encode(
            batch["chunk"],
            show_progress_bar=False,
        )
        return batch


def main() -> None:
    ray.init(ignore_reinit_error=True)
    ray.get([warmup.remote() for _ in range(64)])

    start = time.time()
    ds = ray.data.read_parquet(str(INPUT_PATH))
    if INPUT_LIMIT > 0:
        ds = ds.limit(INPUT_LIMIT)
    ds = ds.filter(lambda row: row["file_name"].endswith(".pdf"))
    ds = ds.map(to_local_pdf_path)
    ds = ds.flat_map(extract_text_from_pdf)
    ds = ds.flat_map(chunker)
    ds = ds.map_batches(
        Embedder,
        batch_size=BATCH_SIZE,
        concurrency=NUM_GPU_NODES,
        num_gpus=1.0,
    )
    ds = ds.select_columns(["uploaded_pdf_path", "page_number", "chunk_id", "chunk", "embedding"])
    ds.write_parquet(str(OUTPUT_PATH))

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
