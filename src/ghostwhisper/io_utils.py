"""Portable audio IO and manifests shared by the artifact commands."""
from pathlib import Path
import csv
import hashlib
import numpy as np
import librosa
import soundfile as sf


def read_audio(path, sample_rate=48000, normalize=True):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    if sr != sample_rate:
        x = librosa.resample(x, orig_sr=sr, target_sr=sample_rate)
    peak = np.max(np.abs(x))
    if normalize and peak > 1e-9:
        x = x / peak * 0.95
    return x.astype(np.float32)


def write_audio(path, x, sample_rate=48000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(x).all():
        raise ValueError("Refusing to save non-finite audio")
    sf.write(path, x, sample_rate, subtype="FLOAT")


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for row in rows:
        sid = row["sample_id"]
        if not sid or Path(sid).name != sid or sid in {".", ".."} or sid in seen:
            raise ValueError(f"Unsafe or duplicate sample_id: {sid}")
        seen.add(sid)
        for key in ("raw_audio", "reference_audio"):
            if row.get(key):
                row[key] = str((path.parent / row[key]).resolve())
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
