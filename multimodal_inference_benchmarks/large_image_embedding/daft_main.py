# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import daft
import numpy as np
import ray
import torch
from daft import DataType, udf
from pybase64 import b64decode
from transformers import ViTForImageClassification, ViTImageProcessor

DATA_ROOT = Path("/data/multimodal_inference_benchmarks/large_image_embedding")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "parquet")).expanduser()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", f"/tmp/daft_large_image_{uuid.uuid4().hex}")).expanduser()
MODEL_ID = os.environ.get("VIT_MODEL", "google/vit-base-patch16-224")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1024"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


def _parquet_input(path: Path) -> str:
    return str(path / "**/*.parquet") if path.is_dir() else str(path)


def _model_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, local_files_only=True)


PROCESSOR = ViTImageProcessor(
    do_convert_rgb=None,
    do_normalize=True,
    do_rescale=True,
    do_resize=True,
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
    resample=2,
    rescale_factor=0.00392156862745098,
    size={"height": 224, "width": 224},
)


@ray.remote
def warmup():
    pass


def decode(data: bytes) -> bytes:
    return b64decode(data, None, True)


def preprocess(image):
    outputs = PROCESSOR(images=image)["pixel_values"]
    if len(outputs) != 1:
        raise ValueError(f"Expected one image, got {len(outputs)}")
    return outputs[0]


@udf(
    return_dtype=DataType.tensor(DataType.float32()),
    batch_size=BATCH_SIZE,
    num_gpus=1,
    concurrency=NUM_GPU_NODES,
)
class Infer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(
            _model_path(),
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def __call__(self, image_column) -> np.ndarray:
        images = torch.from_numpy(np.asarray(image_column.to_pylist())).to(
            dtype=torch.float32,
            device=self.device,
            non_blocking=True,
        )
        with torch.inference_mode():
            return self.model(images).logits.cpu().numpy()


def main() -> None:
    daft.context.set_runner_ray()
    ray.get([warmup.remote() for _ in range(64)])

    start = time.time()
    df = daft.read_parquet(_parquet_input(INPUT_PATH))
    df = df.with_column("image", df["image"].apply(decode, return_dtype=DataType.binary()))
    df = df.with_column("image", df["image"].image.decode(mode=daft.ImageMode.RGB))
    df = df.with_column("height", df["image"].image_height())
    df = df.with_column("width", df["image"].image.width())
    df = df.with_column(
        "image",
        df["image"].apply(preprocess, return_dtype=DataType.tensor(DataType.float32())),
    )
    df = df.with_column("embeddings", Infer(df["image"]))
    df = df.select("embeddings")
    df.write_parquet(str(OUTPUT_PATH))

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
