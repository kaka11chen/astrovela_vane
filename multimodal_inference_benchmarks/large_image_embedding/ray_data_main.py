# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from PIL import Image
from pybase64 import b64decode
from transformers import ViTForImageClassification, ViTImageProcessor

DATA_ROOT = Path("/data/multimodal_inference_benchmarks/large_image_embedding")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "parquet")).expanduser()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", f"/tmp/ray_data_large_image_{uuid.uuid4().hex}")).expanduser()
MODEL_ID = os.environ.get("VIT_MODEL", "google/vit-base-patch16-224")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1024"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


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


def decode(row: dict[str, Any]) -> dict[str, Any]:
    image_data = b64decode(row["image"], None, True)
    with Image.open(BytesIO(image_data)) as image:
        image = image.convert("RGB")
        width, height = image.size
        array = np.asarray(image)
    return {
        "original_url": row["url"],
        "original_width": width,
        "original_height": height,
        "image": array,
    }


def preprocess(row: dict[str, Any]) -> dict[str, Any]:
    outputs = PROCESSOR(images=row["image"])["pixel_values"]
    if len(outputs) != 1:
        raise ValueError(f"Expected one image, got {len(outputs)}")
    row["image"] = outputs[0]
    return row


class Infer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(
            _model_path(),
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        images = torch.from_numpy(batch["image"]).to(
            dtype=torch.float32,
            device=self.device,
            non_blocking=True,
        )
        with torch.inference_mode():
            output = self.model(images).logits.cpu().numpy()
        return {
            "original_url": batch["original_url"],
            "original_width": batch["original_width"],
            "original_height": batch["original_height"],
            "output": output,
        }


def main() -> None:
    ray.init(ignore_reinit_error=True)
    ray.get([warmup.remote() for _ in range(64)])

    start = time.time()
    ds = ray.data.read_parquet(str(INPUT_PATH))
    ds = ds.map(decode).map(preprocess)
    ds = ds.map_batches(
        Infer,
        batch_size=BATCH_SIZE,
        num_gpus=1,
        concurrency=NUM_GPU_NODES,
    )
    ds.write_parquet(str(OUTPUT_PATH))

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
