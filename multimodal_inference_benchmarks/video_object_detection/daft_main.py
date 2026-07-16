# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

# Adapted from Eventual-Inc/Daft's video object detection benchmark.
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import daft
import torch
from daft.expressions import col
from ultralytics import YOLO
from video_kernels import frames_to_torch_tensor, yolo_result_to_features

INPUT_PATH = Path(
    os.environ.get(
        "INPUT_PATH",
        "/data/multimodal_inference_benchmarks/hollywood2/AVIClips",
    )
).expanduser()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", f"/tmp/daft_video_{uuid.uuid4().hex}")).expanduser()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))
INPUT_LIMIT = int(os.environ.get("INPUT_LIMIT", "0"))
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", "640"))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", "640"))
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolo11n.pt")

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


@daft.udf(
    return_dtype=daft.DataType.list(
        daft.DataType.struct(
            {
                "label": daft.DataType.string(),
                "confidence": daft.DataType.float32(),
                "bbox": daft.DataType.list(daft.DataType.int32()),
            }
        )
    ),
    concurrency=NUM_GPU_NODES,
    num_gpus=1.0,
    batch_size=BATCH_SIZE,
)
class ExtractImageFeatures:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        if torch.cuda.is_available():
            self.model.to("cuda")

    def __call__(self, images):
        if len(images) == 0:
            return []
        tensor = frames_to_torch_tensor(images, None)
        return daft.Series.from_pylist(
            [yolo_result_to_features(result) for result in self.model(tensor, verbose=False)]
        )


def main() -> None:
    daft.context.set_runner_ray()

    start = time.time()
    df = daft.read_video_frames(
        str(INPUT_PATH),
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
    )
    if INPUT_LIMIT > 0:
        df = df.limit(INPUT_LIMIT)
    df = df.with_column("features", ExtractImageFeatures(col("data")))
    df = df.explode("features")
    df = df.with_column(
        "object",
        col("data").image.crop(col("features")["bbox"]).image.encode("png"),
    )
    df = df.exclude("data")
    df.write_parquet(str(OUTPUT_PATH))

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
