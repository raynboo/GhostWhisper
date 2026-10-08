"""Checkpoint-driven restoration; no reference speech or transcript used at inference."""
from __future__ import annotations
from pathlib import Path
import argparse
import librosa
import numpy as np
import torch
from .audio import trim_or_pad
from .mel import LogMelTransform, _mel_filterbank
from .models import build_mel_denoise_model
from .stft_model import STFTUNet, compress_spec, decompress_spec, input_features, istft, stft
from .bounded_refiner import refine_waveform

class ResUNetRestorer:
    def __init__(self, checkpoint_path: Path, args: argparse.Namespace, device: torch.device) -> None:
        # Load the full training checkpoint on CPU so optimizer tensors do not
        # consume MPS/CUDA memory. Only the model weights are moved below.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", {})
        model_arch = str(config.get("model_arch", "resunet"))
        base_channels = int(config.get("base_channels", args.resunet_base_channels))
        self.sample_rate = int(config.get("sample_rate", args.sample_rate))
        self.chunk_length = int(round(float(config.get("clip_seconds", args.resunet_chunk_seconds)) * self.sample_rate))
        self.n_fft = int(config.get("n_fft", 4096))
        self.hop_length = int(config.get("hop_length", 512))
        self.n_mels = int(config.get("n_mels", 256))
        self.fmin = float(config.get("fmin", 20.0))
        self.fmax = float(config.get("fmax", 8000.0))
        self.device = device
        self.model = build_mel_denoise_model(model_arch, base_channels=base_channels).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        del checkpoint
        # FFT/complex operations are more portable on CPU. The neural network
        # itself still runs on MPS when selected on Apple Silicon.
        self.feature_device = torch.device("cpu") if device.type == "mps" else device
        self.mel_transform = LogMelTransform(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            db_min=float(config.get("db_min", -80.0)),
            db_max=float(config.get("db_max", 0.0)),
            relative_ref=bool(config.get("relative_ref", True)),
        ).to(self.feature_device)
        filters = _mel_filterbank(self.sample_rate, self.n_fft, self.n_mels, self.fmin, self.fmax).cpu()
        eye = torch.eye(filters.shape[0], dtype=filters.dtype)
        self.inverse_filters = filters.T @ torch.linalg.inv(filters @ filters.T + 1e-4 * eye)
        self.window = torch.hann_window(self.n_fft)

    def mel_db_to_audio(self, mel_db: np.ndarray, target_length: int) -> np.ndarray:
        mel_power = torch.from_numpy(librosa.db_to_power(mel_db, ref=1.0).astype(np.float32))
        linear_power = (self.inverse_filters @ mel_power).clamp_min(0.0)
        magnitude = torch.sqrt(linear_power)
        generator = torch.Generator(device="cpu").manual_seed(1234)
        angles = torch.rand(magnitude.shape, generator=generator) * (2.0 * torch.pi)
        complex_spec = torch.polar(magnitude, angles)
        for _ in range(16):
            audio = torch.istft(complex_spec, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, length=target_length)
            rebuilt = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True, return_complex=True)
            phase = rebuilt / torch.clamp(rebuilt.abs(), min=1e-8)
            complex_spec = magnitude.to(torch.complex64) * phase
        audio = torch.istft(complex_spec, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, length=target_length)
        out = audio.detach().cpu().numpy().astype(np.float32)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1e-9:
            out = out / peak * 0.95
        return out

    def restore(self, em_audio: np.ndarray) -> np.ndarray:
        restored_chunks: list[np.ndarray] = []
        for start in range(0, max(len(em_audio), 1), self.chunk_length):
            chunk = em_audio[start : start + self.chunk_length]
            original_length = len(chunk)
            chunk = trim_or_pad(chunk, self.chunk_length)
            wave = torch.from_numpy(chunk).unsqueeze(0).to(self.feature_device)
            with torch.inference_mode():
                input_mel = self.mel_transform(wave).to(self.device)
                pred_mel = self.model(input_mel).to(self.feature_device)
                pred_db = self.mel_transform.denormalize_db(pred_mel)[0].cpu().numpy()
            restored_chunks.append(self.mel_db_to_audio(pred_db, self.chunk_length)[:original_length])
        return np.concatenate(restored_chunks).astype(np.float32) if restored_chunks else np.zeros(0, dtype=np.float32)


class RefinerRestorer:
    def __init__(self, checkpoint_path: Path, args: argparse.Namespace, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", {})
        self.sample_rate = int(config.get("sample_rate", args.sample_rate))
        self.clip_length = int(round(float(config.get("clip_seconds", args.refiner_clip_seconds)) * self.sample_rate))
        self.n_fft = int(config.get("n_fft", 1024))
        self.hop_length = int(config.get("hop_length", 256))
        self.compress_power = float(config.get("compress_power", 0.30))
        self.output_mode = str(config.get("output_mode", "residual"))
        self.residual_scale = float(config.get("residual_scale", 0.25))
        self.device = device
        self.model = STFTUNet(base_channels=int(config.get("base_channels", 64)), depth=int(config.get("depth", 5)), dropout=0.0).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        del checkpoint

    def restore_chunk(self, noisy: np.ndarray, original_length: int) -> np.ndarray:
        chunk = trim_or_pad(noisy, self.clip_length)
        # Keep complex STFT/ISTFT tensors on CPU; only the real-valued U-Net
        # activations move to MPS. This avoids unsupported MPS complex kernels.
        spectral_device = torch.device("cpu") if self.device.type == "mps" else self.device
        wave = torch.from_numpy(chunk).unsqueeze(0).to(spectral_device)
        if self.output_mode == 'bounded_waveform':
            peak = wave.abs().amax().clamp_min(1e-9)
            with torch.inference_mode():
                pred, _ = refine_waveform(self.model, wave / peak, model_device=self.device,
                                         n_fft=self.n_fft, hop_length=self.hop_length,
                                         compress_power=self.compress_power, residual_scale=self.residual_scale)
            return (pred[0] * peak).cpu().numpy().astype(np.float32)[:original_length]
        with torch.inference_mode():
            noisy_spec = stft(wave, self.n_fft, self.hop_length)
            model_input = input_features(noisy_spec, self.compress_power).to(self.device)
            model_out = self.model(model_input).to(spectral_device)
            if self.output_mode == "residual":
                pred_comp = compress_spec(noisy_spec, self.compress_power) + self.residual_scale * model_out
            else:
                pred_comp = model_out
            pred_spec = decompress_spec(pred_comp, self.compress_power)
            pred_wave = istft(pred_spec, self.clip_length, self.n_fft, self.hop_length)
        out = pred_wave[0].detach().cpu().numpy().astype(np.float32)[:original_length]
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1e-9:
            out = out / peak * 0.95
        return out

    def restore(self, noisy_audio: np.ndarray) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, max(len(noisy_audio), 1), self.clip_length):
            chunk = noisy_audio[start : start + self.clip_length]
            chunks.append(self.restore_chunk(chunk, len(chunk)))
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
