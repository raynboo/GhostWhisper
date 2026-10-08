"""Paper-sized restoration networks; extracted from the research training code."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Dropout2d(dropout),
            group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class STFTUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, depth: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.depth = depth
        channels = [base_channels * (2**idx) for idx in range(depth)]
        self.input = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for idx, channel in enumerate(channels):
            self.down_blocks.append(nn.Sequential(ResBlock(channel, dropout), ResBlock(channel, dropout)))
            if idx < depth - 1:
                self.downsample.append(nn.Conv2d(channel, channels[idx + 1], 4, stride=2, padding=1))
        self.mid = nn.Sequential(ResBlock(channels[-1], dropout), ResBlock(channels[-1], dropout))
        self.upsample = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for idx in range(depth - 2, -1, -1):
            self.upsample.append(nn.Conv2d(channels[idx + 1], channels[idx], 3, padding=1))
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[idx] * 2, channels[idx], 3, padding=1),
                    ResBlock(channels[idx], dropout),
                    ResBlock(channels[idx], dropout),
                )
            )
        self.output = nn.Sequential(group_norm(channels[0]), nn.SiLU(), nn.Conv2d(channels[0], 2, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        multiple = 2 ** (self.depth - 1)
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.input(x)
        skips = []
        for idx, block in enumerate(self.down_blocks):
            x = block(x)
            skips.append(x)
            if idx < len(self.downsample):
                x = self.downsample[idx](x)
        x = self.mid(x)
        for up, block, skip in zip(self.upsample, self.up_blocks, reversed(skips[:-1]), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(x)
            x = block(torch.cat([x, skip], dim=1))
        x = self.output(x)
        return x[:, :, :height, :width]


def stft(wave: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=wave.device, dtype=wave.dtype)
    return torch.stft(wave, n_fft=n_fft, hop_length=hop_length, window=window, center=True, return_complex=True)


def istft(spec: torch.Tensor, length: int, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=spec.device, dtype=torch.float32)
    return torch.istft(spec.to(torch.complex64), n_fft=n_fft, hop_length=hop_length, window=window, center=True, length=length)


def compress_spec(spec: torch.Tensor, power: float) -> torch.Tensor:
    mag = spec.abs().clamp_min(1e-8)
    comp = (mag**power) * (spec / mag)
    return torch.stack([comp.real, comp.imag], dim=1)


def decompress_spec(comp: torch.Tensor, power: float) -> torch.Tensor:
    real = comp[:, 0]
    imag = comp[:, 1]
    mag_c = torch.sqrt(real.square() + imag.square()).clamp_min(1e-8)
    mag = mag_c ** (1.0 / power)
    phase = torch.complex(real / mag_c, imag / mag_c)
    return phase * mag


def input_features(noisy_spec: torch.Tensor, power: float) -> torch.Tensor:
    comp = compress_spec(noisy_spec, power)
    logmag = torch.log1p(noisy_spec.abs())
    scale = logmag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    logmag = (logmag / scale).unsqueeze(1)
    return torch.cat([comp, logmag], dim=1)
