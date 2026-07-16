# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path

import numpy as np
import ray
import torch
from packaging import version
from PIL import Image
from ray.data.expressions import download
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

DATA_ROOT = Path("/data/multimodal_inference_benchmarks/imagenet")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "metadata_file.parquet")).expanduser()
IMAGE_ROOT = Path(os.environ.get("LOCAL_IMAGE_ROOT", DATA_ROOT / "train")).expanduser()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", f"/tmp/ray_data_image_{uuid.uuid4().hex}")).expanduser()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))
INPUT_LIMIT = int(os.environ.get("INPUT_LIMIT", "803580"))

WEIGHTS = ResNet18_Weights.DEFAULT
TRANSFORM = transforms.Compose([transforms.ToTensor(), WEIGHTS.transforms()])

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


def _local_image_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    normalized = value.replace("\\", "/")
    marker = "imagenet/train/"
    relative = normalized.split(marker, 1)[1] if marker in normalized else normalized.lstrip("/")
    return IMAGE_ROOT / relative


@ray.remote
def warmup():
    pass


def to_local_image_path(row):
    row["image_url"] = str(_local_image_path(row["image_url"]))
    return row


def deserialize_image(row):
    with Image.open(io.BytesIO(row.pop("bytes"))) as image:
        row["image"] = np.asarray(image.convert("RGB"))
    return row


def transform_image(row):
    row["norm_image"] = TRANSFORM(row.pop("image")).numpy()
    return row


class ResNetActor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(weights=WEIGHTS).to(self.device)
        self.model.eval()

    def __call__(self, batch):
        images = torch.from_numpy(batch.pop("norm_image")).to(self.device)
        with torch.inference_mode():
            classes = self.model(images).argmax(dim=1).cpu()
        batch["label"] = [WEIGHTS.meta["categories"][index] for index in classes]
        return batch


def _classify(ds, path_column: str):
    return ds.map_batches(
        ResNetActor,
        batch_size=BATCH_SIZE,
        num_gpus=1.0,
        concurrency=NUM_GPU_NODES,
    ).select_columns([path_column, "label"])


def main() -> None:
    ray.init(ignore_reinit_error=True)
    ray.get([warmup.remote() for _ in range(64)])

    start = time.time()
    metadata = ray.data.read_parquet(str(INPUT_PATH))
    if INPUT_LIMIT > 0:
        metadata = metadata.limit(INPUT_LIMIT)

    # Ray 2.50 introduced the expression-based local-file download path.
    if version.parse(ray.__version__) > version.parse("2.49.2"):
        ds = metadata.map(to_local_image_path)
        ds = ds.with_column("bytes", download("image_url"))
        ds = ds.map(deserialize_image).map(transform_image)
        ds = _classify(ds, "image_url")
    else:
        paths = [row["image_url"] for row in metadata.map(to_local_image_path).take_all()]
        ds = ray.data.read_images(
            paths,
            include_paths=True,
            ignore_missing_paths=True,
            mode="RGB",
        )
        ds = _classify(ds.map(transform_image), "path")

    ds.write_parquet(str(OUTPUT_PATH))
    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
