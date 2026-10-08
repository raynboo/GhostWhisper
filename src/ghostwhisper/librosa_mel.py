from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_N_FFT = 4096
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 256
DEFAULT_FMIN = 20.0
DEFAULT_FMAX = 8000.0
DEFAULT_DB_MIN = -80.0
DEFAULT_DB_MAX = 0.0


def compute_librosa_mel_db(
    samples: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
) -> np.ndarray:
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


def save_librosa_mel_figure(
    path: Path,
    mel_db: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    db_min: float = DEFAULT_DB_MIN,
    db_max: float = DEFAULT_DB_MAX,
    title: str = "Mel spectrogram",
    figsize: tuple[float, float] = (12.0, 5.0),
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    image = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        cmap="magma",
        vmin=db_min,
        vmax=db_max,
        ax=ax,
    )
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_librosa_mel_panels(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    db_min: float = DEFAULT_DB_MIN,
    db_max: float = DEFAULT_DB_MAX,
    title: str = "Mel comparison",
    figsize_per_panel: tuple[float, float] = (12.0, 3.6),
    dpi: int = 150,
) -> None:
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * len(panels)),
        sharex=True,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]
    image = None
    for ax, (panel_title, mel_db) in zip(axes, panels, strict=True):
        image = librosa.display.specshow(
            mel_db,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            fmin=fmin,
            fmax=fmax,
            cmap="magma",
            vmin=db_min,
            vmax=db_max,
            ax=ax,
        )
        ax.set_title(panel_title)
    fig.suptitle(title)
    if image is not None:
        fig.colorbar(image, ax=axes, format="%+2.0f dB", fraction=0.025, pad=0.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
