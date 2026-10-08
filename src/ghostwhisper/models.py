from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MelDenoiseUNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        channels = base_channels
        self.enc1 = ConvBlock(1, channels)
        self.enc2 = ConvBlock(channels, channels * 2)
        self.enc3 = ConvBlock(channels * 2, channels * 4)
        self.bottleneck = ConvBlock(channels * 4, channels * 8)

        self.down = nn.AvgPool2d(kernel_size=2)
        self.up3 = nn.Conv2d(channels * 8, channels * 4, kernel_size=1)
        self.dec3 = ConvBlock(channels * 8, channels * 4)
        self.up2 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1)
        self.dec2 = ConvBlock(channels * 4, channels * 2)
        self.up1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.dec1 = ConvBlock(channels * 2, channels)
        self.out = nn.Conv2d(channels, 1, kernel_size=1)

    def _up_to(self, x: torch.Tensor, skip: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return projection(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        b = self.bottleneck(self.down(e3))

        d3 = self._up_to(b, e3, self.up3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self._up_to(d3, e2, self.up2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._up_to(d2, e1, self.up1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        residual = torch.tanh(self.out(d1))
        return torch.clamp(x + residual, -1.0, 1.0)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.transpose(1, 2).reshape(batch, channels, height, width)
        return x + h


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        dropout: float,
        use_attention: bool,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ResidualBlock2D(in_channels, out_channels, dropout)]
        blocks.extend(ResidualBlock2D(out_channels, out_channels, dropout) for _ in range(num_blocks - 1))
        if use_attention:
            blocks.append(SelfAttention2D(out_channels))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_blocks: int,
        dropout: float,
        use_attention: bool,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ResidualBlock2D(in_channels + skip_channels, out_channels, dropout)]
        blocks.extend(ResidualBlock2D(out_channels, out_channels, dropout) for _ in range(num_blocks - 1))
        if use_attention:
            blocks.append(SelfAttention2D(out_channels))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat([x, skip], dim=1))


class MelDenoiseResUNet(nn.Module):
    """A stronger residual U-Net for direct log-mel restoration."""

    def __init__(
        self,
        base_channels: int = 48,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_blocks: int = 2,
        dropout: float = 0.05,
        attention_levels: tuple[int, ...] = (),
        bottleneck_attention: bool = False,
        residual_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        channels = [base_channels * mult for mult in channel_mult]

        self.input_proj = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels = channels[0]
        for level, out_channels in enumerate(channels):
            self.down_blocks.append(
                DownBlock(
                    in_channels,
                    out_channels,
                    num_blocks=num_blocks,
                    dropout=dropout,
                    use_attention=level in attention_levels,
                )
            )
            if level < len(channels) - 1:
                self.downsamples.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                )
            in_channels = out_channels

        bottleneck_channels = channels[-1]
        bottleneck_blocks: list[nn.Module] = [ResidualBlock2D(bottleneck_channels, bottleneck_channels, dropout)]
        if bottleneck_attention:
            bottleneck_blocks.append(SelfAttention2D(bottleneck_channels))
        bottleneck_blocks.append(ResidualBlock2D(bottleneck_channels, bottleneck_channels, dropout))
        self.bottleneck = nn.Sequential(*bottleneck_blocks)

        self.up_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        for level in reversed(range(len(channels) - 1)):
            self.up_blocks.append(
                UpBlock(
                    current_channels,
                    skip_channels=channels[level],
                    out_channels=channels[level],
                    num_blocks=num_blocks,
                    dropout=dropout,
                    use_attention=level in attention_levels,
                )
            )
            current_channels = channels[level]

        self.output_norm = nn.GroupNorm(_group_count(current_channels), current_channels)
        self.output_proj = nn.Conv2d(current_channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        skips: list[torch.Tensor] = []
        for level, block in enumerate(self.down_blocks):
            h = block(h)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self.bottleneck(h)
        for block, skip in zip(self.up_blocks, reversed(skips[:-1]), strict=True):
            h = block(h, skip)

        residual = torch.tanh(self.output_proj(F.silu(self.output_norm(h))))
        return torch.clamp(x + self.residual_scale * residual, -1.0, 1.0)


def build_mel_denoise_model(
    arch: str = "resunet",
    base_channels: int = 64,
) -> nn.Module:
    if arch == "unet":
        return MelDenoiseUNet(base_channels=base_channels)
    if arch == "resunet":
        return MelDenoiseResUNet(base_channels=base_channels)
    raise ValueError(f"Unknown mel denoise architecture: {arch}")
