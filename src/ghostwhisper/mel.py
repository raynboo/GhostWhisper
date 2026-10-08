from __future__ import annotations

import torch
from torch import nn


def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    mel_edges = torch.linspace(_hz_to_mel(torch.tensor(fmin)), _hz_to_mel(torch.tensor(fmax)), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    bins = torch.floor((n_fft + 1) * hz_edges / sample_rate).long().clamp(0, n_fft // 2)

    filters = torch.zeros(n_mels, n_fft // 2 + 1, dtype=torch.float32)
    for idx in range(n_mels):
        left, center, right = int(bins[idx]), int(bins[idx + 1]), int(bins[idx + 2])
        if center > left:
            filters[idx, left:center] = (torch.arange(left, center) - left) / (center - left)
        if right > center:
            filters[idx, center:right] = (right - torch.arange(center, right)) / (right - center)
    return filters


class LogMelTransform(nn.Module):
    def __init__(
        self,
        sample_rate: int = 48_000,
        n_fft: int = 2048,
        hop_length: int = 480,
        n_mels: int = 96,
        fmin: float = 40.0,
        fmax: float = 12_000.0,
        db_min: float = -100.0,
        db_max: float = -20.0,
        relative_ref: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.db_min = db_min
        self.db_max = db_max
        self.relative_ref = relative_ref
        filters = _mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
        self.register_buffer("mel_filters", filters)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = stft.abs().pow(2)
        mel = torch.einsum("mf,bft->bmt", self.mel_filters, power)
        log_mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        if self.relative_ref:
            log_mel = log_mel - torch.amax(log_mel, dim=(-2, -1), keepdim=True)
        normalized = (torch.clamp(log_mel, self.db_min, self.db_max) - self.db_min)
        normalized = normalized / (self.db_max - self.db_min)
        return normalized.mul(2.0).sub(1.0).unsqueeze(1)

    def denormalize_db(self, normalized: torch.Tensor) -> torch.Tensor:
        values = normalized.squeeze(1).add(1.0).mul(0.5)
        return values * (self.db_max - self.db_min) + self.db_min
