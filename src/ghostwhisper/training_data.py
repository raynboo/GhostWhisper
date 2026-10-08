from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def read_pair_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ('clean_audio_path', 'noisy_audio_path'):
            value = Path(row[key])
            row[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return rows


def split_rows_by_clean_id(
    rows: list[dict[str, str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if any(row.get('split') for row in rows):
        if any(row.get('split') not in {'train', 'val'} for row in rows):
            raise ValueError('Training manifest must use only explicit train/val splits; exclude test rows')
        train_rows = [r for r in rows if r['split'] == 'train']
        val_rows = [r for r in rows if r['split'] == 'val']
        for key in ('clean_clip_id', 'speaker_id'):
            a = {r[key] for r in train_rows if r.get(key)}
            b = {r[key] for r in val_rows if r.get(key)}
            if a & b:
                raise ValueError(f'Train/validation overlap in {key}')
        if not train_rows or not val_rows:
            raise ValueError('Both train and validation data are required')
        return train_rows, val_rows
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    threshold = int(val_ratio * 10_000)
    for row in rows:
        key = f"{seed}:{row['clean_clip_id']}"
        bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 10_000
        if bucket < threshold:
            val_rows.append(row)
        else:
            train_rows.append(row)
    if not train_rows or not val_rows:
        raise ValueError('Empty training/validation split; provide an explicit split column')
    return train_rows, val_rows


def _load_wav_mono(path: Path, target_length: int | None = None, sample_rate: int = 48000) -> np.ndarray:
    samples, sr = sf.read(path)
    if sr != sample_rate:
        raise ValueError(f'{path}: expected {sample_rate} Hz, got {sr}; resample before training')
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    samples = samples.astype(np.float32)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f'{path}: empty or non-finite audio')
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples / peak
    if target_length is not None:
        if len(samples) >= target_length:
            samples = samples[:target_length]
        else:
            padded = np.zeros(target_length, dtype=np.float32)
            padded[: len(samples)] = samples
            samples = padded
    return samples.astype(np.float32)


class PairWaveformDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        sample_rate: int = 48_000,
        clip_seconds: float = 4.0,
    ) -> None:
        self.rows = rows
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * clip_seconds)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        noisy = _load_wav_mono(Path(row["noisy_audio_path"]), self.target_length, self.sample_rate)
        clean = _load_wav_mono(Path(row["clean_audio_path"]), self.target_length, self.sample_rate)
        return {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean),
            "pair_id": row["pair_id"],
            "transcript": row.get("transcript", ""),
        }
