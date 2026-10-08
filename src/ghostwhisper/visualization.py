from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MPL_CACHE_DIR = Path("artifacts/.mpl_cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))

import matplotlib.pyplot as plt
import librosa
import librosa.display


def log_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    n_fft: int = 4096,
    hop_length: int = 512,
    n_mels: int = 256,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    fmax = fmax or min(2000.0, sample_rate / 2.0)
    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def save_pair_mel_figure(
    clean: np.ndarray,
    noisy: np.ndarray,
    sample_rate: int,
    output_path: Path,
    title: str,
) -> None:
    hop_length = 512
    fmin = 20.0
    fmax = min(2000.0, sample_rate / 2.0)
    clean_mel = log_mel_spectrogram(clean, sample_rate, hop_length=hop_length, fmin=fmin, fmax=fmax)
    noisy_mel = log_mel_spectrogram(noisy, sample_rate, hop_length=hop_length, fmin=fmin, fmax=fmax)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for ax, data, label in [
        (axes[0], clean_mel, "Clean 48 kHz clip"),
        (axes[1], noisy_mel, "Synthetic noisy clip"),
    ]:
        image = librosa.display.specshow(
            data,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            fmin=fmin,
            fmax=fmax,
            cmap="magma",
            vmin=-80.0,
            vmax=0.0,
            ax=ax,
        )
        ax.set_title(label)
        fig.colorbar(image, ax=ax, format="%+2.0f dB", pad=0.01)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
