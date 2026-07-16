# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

# Adapted from Eventual-Inc/Daft's image classification benchmark.
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import daft
import numpy as np
import ray
import torch
from daft import col
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

DATA_ROOT = Path("/data/multimodal_inference_benchmarks/imagenet")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "metadata_file.parquet")).expanduser()
IMAGE_ROOT = Path(os.environ.get("LOCAL_IMAGE_ROOT", DATA_ROOT / "train")).expanduser()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", f"/tmp/daft_image_{uuid.uuid4().hex}")).expanduser()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))
INPUT_LIMIT = int(os.environ.get("INPUT_LIMIT", "803580"))

IMAGE_SHAPE = (3, 224, 224)
WEIGHTS = ResNet18_Weights.DEFAULT
TRANSFORM = transforms.Compose([transforms.ToTensor(), WEIGHTS.transforms()])

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


def _parquet_input(path: Path) -> str:
    return str(path / "**/*.parquet") if path.is_dir() else str(path)


def _local_image_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return str(path)
    normalized = value.replace("\\", "/")
    marker = "imagenet/train/"
    relative = normalized.split(marker, 1)[1] if marker in normalized else normalized.lstrip("/")
    return str(IMAGE_ROOT / relative)


def transform_image(image):
    return TRANSFORM(image)


@ray.remote
def warmup():
    pass


@daft.udf(
    return_dtype=daft.DataType.string(),
    concurrency=NUM_GPU_NODES,
    num_gpus=1.0,
    batch_size=BATCH_SIZE,
)
class ResNetModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(weights=WEIGHTS).to(self.device)
        self.model.eval()

    def __call__(self, images):
        if len(images) == 0:
            return []
        batch = torch.from_numpy(np.asarray(images.to_pylist())).to(self.device)
        with torch.inference_mode():
            classes = self.model(batch).argmax(dim=1).cpu()
        return [WEIGHTS.meta["categories"][index] for index in classes]


def main() -> None:
    daft.context.set_runner_ray()
    ray.get([warmup.remote() for _ in range(64)])

    start = time.time()
    df = daft.read_parquet(_parquet_input(INPUT_PATH))
    if INPUT_LIMIT > 0:
        df = df.limit(INPUT_LIMIT)
    df = df.with_column(
        "image_url",
        df["image_url"].apply(_local_image_path, return_dtype=daft.DataType.string()),
    )

    # Matching GPU-count partitions avoids tiny actor batches on this metadata file.
    df = df.repartition(NUM_GPU_NODES)
    df = df.with_column(
        "decoded_image",
        df["image_url"]
        .url.download()
        .image.decode(
            on_error="null",
            mode=daft.ImageMode.RGB,
        ),
    )
    df = df.where(df["decoded_image"].not_null())
    df = df.with_column(
        "norm_image",
        df["decoded_image"].apply(
            transform_image,
            return_dtype=daft.DataType.tensor(
                dtype=daft.DataType.float32(),
                shape=IMAGE_SHAPE,
            ),
        ),
    )
    df = df.with_column("label", ResNetModel(col("norm_image")))
    df = df.select("image_url", "label")
    df.write_parquet(str(OUTPUT_PATH))

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
