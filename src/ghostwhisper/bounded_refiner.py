"""Bounded waveform residual using the paper-sized STFT U-Net (no extra weights)."""
import torch
from .stft_model import stft, istft, input_features


def refine_waveform(model, wave, *, model_device, n_fft=1024, hop_length=256,
                    compress_power=.3, residual_scale=.1):
    """The complex transforms stay on CPU for MPS; transfers preserve gradients.

    The network predicts a complex residual spectrum; inverse STFT followed by
    tanh bounds each waveform correction to +/- residual_scale. This is a NEW
    artifact-training variant, not a reinterpretation of the historical weights.
    """
    spectral_device = torch.device('cpu') if torch.device(model_device).type == 'mps' else torch.device(model_device)
    wave = wave.to(spectral_device)
    spec = stft(wave, n_fft, hop_length)
    out = model(input_features(spec, compress_power).to(model_device)).to(spectral_device)
    correction_spec = torch.complex(out[:, 0], out[:, 1])
    correction = istft(correction_spec, wave.shape[-1], n_fft, hop_length)
    delta = residual_scale * torch.tanh(correction)
    return wave + delta, delta
