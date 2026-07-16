# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

# Adapted from Eventual-Inc/Daft's audio transcription benchmark.
from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path

import daft
import numpy as np
import ray
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

DATA_ROOT = Path("/data/multimodal_inference_benchmarks")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "common_voice_17/parquet")).expanduser()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", f"/tmp/daft_audio_{uuid.uuid4().hex}")).expanduser()
MODEL_ID = os.environ.get("TRANSCRIPTION_MODEL", "openai/whisper-tiny")
SAMPLING_RATE = 16000
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))
INPUT_LIMIT = int(os.environ.get("INPUT_LIMIT", "0"))

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


def _parquet_input(path: Path) -> str:
    return str(path / "**/*.parquet") if path.is_dir() else str(path)


def _model_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, local_files_only=True)


processor = AutoProcessor.from_pretrained(_model_path(), local_files_only=True)


@ray.remote
def warmup():
    pass


def _decode_audio_bytes(payload: bytes):
    for fmt in ("flac", None, "wav", "mp3"):
        try:
            if fmt is None:
                return torchaudio.load(io.BytesIO(payload))
            return torchaudio.load(io.BytesIO(payload), format=fmt)
        except Exception:
            continue
    raise ValueError("Failed to decode audio bytes")


def resample(audio_bytes):
    waveform, sampling_rate = _decode_audio_bytes(audio_bytes)
    waveform = T.Resample(sampling_rate, SAMPLING_RATE)(waveform).squeeze()
    return np.asarray(waveform)


@daft.udf(return_dtype=daft.DataType.tensor(daft.DataType.float32()))
def whisper_preprocess(resampled):
    return processor(
        resampled.to_arrow().to_numpy(zero_copy_only=False).tolist(),
        sampling_rate=SAMPLING_RATE,
    ).input_features


@daft.udf(
    return_dtype=daft.DataType.list(daft.DataType.int32()),
    batch_size=BATCH_SIZE,
    concurrency=NUM_GPU_NODES,
    num_gpus=1,
)
class Transcriber:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            _model_path(),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def __call__(self, extracted_features):
        features = torch.as_tensor(
            np.asarray(extracted_features),
            device=self.device,
            dtype=torch.float16,
        )
        with torch.inference_mode():
            return self.model.generate(features).cpu().numpy()


@daft.udf(return_dtype=daft.DataType.string())
def decoder(token_ids):
    return processor.batch_decode(token_ids, skip_special_tokens=True)


def main() -> None:
    daft.context.set_runner_ray()
    ray.get([warmup.remote() for _ in range(64)])

    start = time.time()
    df = daft.read_parquet(_parquet_input(INPUT_PATH))
    if INPUT_LIMIT > 0:
        df = df.limit(INPUT_LIMIT)
    df = df.with_column(
        "resampled",
        df["audio"]["bytes"].apply(
            resample,
            return_dtype=daft.DataType.list(daft.DataType.float32()),
        ),
    )
    df = df.with_column("extracted_features", whisper_preprocess(df["resampled"]))
    df = df.with_column("token_ids", Transcriber(df["extracted_features"]))
    df = df.with_column("transcription", decoder(df["token_ids"]))
    df = df.with_column("transcription_length", df["transcription"].str.length())
    df = df.exclude("token_ids", "extracted_features", "resampled")
    df.write_parquet(str(OUTPUT_PATH))

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
