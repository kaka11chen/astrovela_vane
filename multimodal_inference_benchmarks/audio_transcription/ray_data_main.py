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
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

DATA_ROOT = Path("/data/multimodal_inference_benchmarks")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "common_voice_17/parquet")).expanduser()
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", f"/tmp/ray_data_audio_{uuid.uuid4().hex}")).expanduser()
MODEL_ID = os.environ.get("TRANSCRIPTION_MODEL", "openai/whisper-tiny")
SAMPLING_RATE = 16000
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))
INPUT_LIMIT = int(os.environ.get("INPUT_LIMIT", "0"))

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


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


def resample(row):
    audio = row.pop("audio")
    waveform, sampling_rate = _decode_audio_bytes(audio["bytes"])
    waveform = T.Resample(sampling_rate, SAMPLING_RATE)(waveform).squeeze()
    row["arr"] = np.asarray(waveform)
    return row


def whisper_preprocess(batch):
    audio = batch.pop("arr")
    features = processor(
        audio.tolist(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="np",
    ).input_features
    batch["input_features"] = list(features)
    return batch


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

    def __call__(self, batch):
        features = torch.as_tensor(
            np.asarray(batch.pop("input_features")),
            device=self.device,
            dtype=torch.float16,
        )
        with torch.inference_mode():
            batch["token_ids"] = self.model.generate(features).cpu().numpy()
        return batch


def decoder(batch):
    token_ids = batch.pop("token_ids")
    transcription = processor.batch_decode(token_ids, skip_special_tokens=True)
    batch["transcription"] = transcription
    batch["transcription_length"] = np.asarray([len(text) for text in transcription])
    return batch


def main() -> None:
    ray.init(ignore_reinit_error=True)
    ray.get([warmup.remote() for _ in range(64)])

    start = time.time()
    ds = ray.data.read_parquet(str(INPUT_PATH))
    if INPUT_LIMIT > 0:
        ds = ds.limit(INPUT_LIMIT)
    ds = ds.repartition(target_num_rows_per_block=BATCH_SIZE)
    ds = ds.map(resample)
    ds = ds.map_batches(whisper_preprocess, batch_size=BATCH_SIZE)
    ds = ds.map_batches(
        Transcriber,
        batch_size=BATCH_SIZE,
        concurrency=NUM_GPU_NODES,
        num_gpus=1,
    )
    ds = ds.map_batches(decoder, batch_size=BATCH_SIZE)
    ds.write_parquet(str(OUTPUT_PATH))

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
