"""Historical metric conventions and RAW-based alignment (not waveform SNR)."""
from __future__ import annotations
import math
import re
import numpy as np
import librosa
from scipy import signal

def mel_db(samples: np.ndarray, sample_rate: int, n_fft: int = 4096, hop_length: int = 512) -> np.ndarray:
    if samples.size == 0 or float(np.max(np.abs(samples))) <= 1e-12:
        return np.full((256, 1), -80.0, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=256,
        fmin=20.0,
        fmax=8000.0,
        power=2.0,
    )
    return np.clip(librosa.power_to_db(mel, ref=np.max), -80.0, 0.0).astype(np.float32)


def log_mag_db(samples: np.ndarray, sample_rate: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    if samples.size == 0:
        samples = np.zeros(1, dtype=np.float32)
    spec = librosa.stft(samples.astype(np.float32), n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    return np.clip(20.0 * np.log10(np.maximum(np.abs(spec), 1e-8)), -120.0, 20.0).astype(np.float32)


def common_frames(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = min(a.shape[-1], b.shape[-1])
    return a[..., :frames], b[..., :frames]


def mel_activity_profile(samples: np.ndarray, sample_rate: int, profile_rate: int = 100) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(1, dtype=np.float32)
    hop_length = max(1, int(round(sample_rate / profile_rate)))
    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32),
        sr=sample_rate,
        n_fft=2048,
        hop_length=hop_length,
        n_mels=96,
        fmin=100.0,
        fmax=min(6000.0, sample_rate / 2.0),
        power=2.0,
    )
    mel_db_values = librosa.power_to_db(mel, ref=np.max)
    # Suppress stationary EM tones and broadband floors before matching time.
    excess = np.maximum(mel_db_values - np.median(mel_db_values, axis=1, keepdims=True), 0.0)
    profile = np.mean(excess, axis=0)
    if len(profile) >= 9:
        window = min(21, len(profile) if len(profile) % 2 else len(profile) - 1)
        if window >= 5:
            profile = signal.savgol_filter(profile, window, min(3, window - 2), mode="interp")
    profile = profile - float(np.mean(profile))
    scale = float(np.std(profile))
    if scale > 1e-8:
        profile = profile / scale
    return profile.astype(np.float32)


def align_for_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    max_shift_sec: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    ref_profile = mel_activity_profile(reference, sample_rate)
    cand_profile = mel_activity_profile(candidate, sample_rate)
    if float(np.std(ref_profile)) < 1e-6 or float(np.std(cand_profile)) < 1e-6:
        ref, cand = apply_alignment_shift(reference, candidate, 0)
        return ref, cand, 0.0
    correlation = signal.correlate(cand_profile, ref_profile, mode="full", method="fft")
    lags = signal.correlation_lags(len(cand_profile), len(ref_profile), mode="full")
    max_shift_frames = int(round(max_shift_sec * 100))
    keep = np.abs(lags) <= max_shift_frames
    lag_frames = int(lags[keep][np.argmax(correlation[keep])]) if np.any(keep) else 0
    lag_samples = int(round(lag_frames * sample_rate / 100))
    ref, cand = apply_alignment_shift(reference, candidate, lag_samples)
    return ref, cand, lag_samples / sample_rate


def apply_alignment_shift(
    reference: np.ndarray,
    candidate: np.ndarray,
    lag_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag_samples > 0:
        length = min(len(reference), len(candidate) - lag_samples)
        ref = reference[: max(0, length)]
        cand = candidate[lag_samples : lag_samples + len(ref)]
    elif lag_samples < 0:
        ref_start = -lag_samples
        length = min(len(reference) - ref_start, len(candidate))
        ref = reference[ref_start : ref_start + max(0, length)]
        cand = candidate[: len(ref)]
    else:
        length = min(len(reference), len(candidate))
        ref = reference[:length]
        cand = candidate[:length]
    return ref.astype(np.float32), cand.astype(np.float32)


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    signal_power = float(np.sum(reference**2))
    noise_power = float(np.sum((candidate - reference) ** 2))
    return 10.0 * math.log10((signal_power + 1e-12) / (noise_power + 1e-12))


def lsd_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref, cand = common_frames(reference, candidate)
    return float(np.sqrt(np.mean((cand - ref) ** 2)))


def optional_stoi_pesq(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> dict[str, float]:
    out = {"stoi": float("nan"), "pesq_wb": float("nan")}
    length = min(len(reference), len(candidate))
    if length < int(0.25 * sample_rate):
        return out
    ref = reference[:length]
    cand = candidate[:length]
    try:
        from pystoi import stoi

        ref_10 = librosa.resample(ref, orig_sr=sample_rate, target_sr=10_000)
        cand_10 = librosa.resample(cand, orig_sr=sample_rate, target_sr=10_000)
        n = min(len(ref_10), len(cand_10))
        out["stoi"] = float(stoi(ref_10[:n], cand_10[:n], 10_000, extended=False))
    except Exception:
        pass
    try:
        from pesq import pesq

        ref_16 = librosa.resample(ref, orig_sr=sample_rate, target_sr=16_000)
        cand_16 = librosa.resample(cand, orig_sr=sample_rate, target_sr=16_000)
        n = min(len(ref_16), len(cand_16))
        out["pesq_wb"] = float(pesq(16_000, ref_16[:n], cand_16[:n], "wb"))
    except Exception:
        pass
    return out


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return float("nan")
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, start=1):
        cur = [i]
        for j, hw in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if rw == hw else 1)))
        prev = cur
    return prev[-1] / max(1, len(ref))
