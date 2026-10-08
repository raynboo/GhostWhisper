#!/usr/bin/env python3
"""Export equivalent reference-plane impedance from OSL-calibrated Gamma.

Added for artifact packaging; this is not an archived paper-results script.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def gamma_to_impedance(gamma, valid, z0=50.0, denominator_floor=1e-6):
    gamma = np.asarray(gamma, dtype=np.complex128)
    valid = np.asarray(valid, dtype=bool).copy()
    if gamma.ndim != 1 or valid.shape != gamma.shape:
        raise ValueError('Gamma and validity mask must be equal-length vectors')
    if not np.isfinite(z0) or z0 <= 0:
        raise ValueError('Reference impedance must be positive and finite')
    if not np.isfinite(denominator_floor) or denominator_floor <= 0:
        raise ValueError('Denominator floor must be positive and finite')
    valid &= np.isfinite(gamma) & (np.abs(1 - gamma) > denominator_floor)
    impedance = np.full(gamma.shape, np.nan + 1j * np.nan)
    impedance[valid] = z0 * (1 + gamma[valid]) / (1 - gamma[valid])
    return impedance, valid


def tone_semiaxes(time_s, impedance, valid, frequency_hz):
    """Fit DC + complex cosine/sine; singular values are ellipse semiaxes."""
    t = np.asarray(time_s, dtype=float)
    z = np.asarray(impedance, dtype=np.complex128)
    valid = np.asarray(valid, dtype=bool)
    if t.shape != z.shape or valid.shape != z.shape or t.ndim != 1:
        raise ValueError('Time, impedance and validity must have equal vector shapes')
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError('Tone frequency must be positive and finite')
    mask = valid & np.isfinite(t) & np.isfinite(z)
    t, z = t[mask], z[mask]
    if len(t) < 4 or np.any(np.diff(t) <= 0):
        raise ValueError('At least four increasing valid timestamps are required')
    if frequency_hz >= 0.5 / np.median(np.diff(t)):
        raise ValueError('Tone must be below the sampled Nyquist frequency')
    phase = 2 * np.pi * frequency_hz * (t - t[0])
    design = np.column_stack([np.ones(len(t)), np.cos(phase), np.sin(phase)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if rank != 3:
        raise ValueError('Tone fit is rank deficient')
    axes = np.linalg.svd(np.array([coefficients[1:].real,
                                  coefficients[1:].imag]), compute_uv=False)
    return {'frequency_hz': float(frequency_hz),
            'major_axis_semiamplitude_ohm': float(axes[0]),
            'minor_axis_semiamplitude_ohm': float(axes[1]),
            'fit_residual_rms_ohm': float(np.sqrt(np.mean(np.abs(z - design @ coefficients)**2))),
            'fit_samples': int(len(t))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('calibrated', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--z0', type=float, default=50.0)
    parser.add_argument('--denominator-floor', type=float, default=1e-6)
    parser.add_argument('--tone-hz', type=float, nargs='+', required=True,
                        help='Frequencies to fit, also for the silent control')
    args = parser.parse_args()
    with h5py.File(args.calibrated, 'r') as handle:
        if handle.attrs.get('format') != 'b210_calibrated_one_port_reflection':
            raise ValueError('Input must be the calibrated.h5 output of osl_calibrate.py apply')
        time_s = handle['processed/time_s'][:]
        gamma = handle['processed/calibrated_gamma'][:]
        valid = handle['processed/valid_calibration'][:]
    impedance, valid = gamma_to_impedance(gamma, valid, args.z0, args.denominator_floor)
    fits = [tone_semiaxes(time_s, impedance, valid, f) for f in args.tone_hz]
    summary = {'source': str(args.calibrated), 'reference_impedance_ohm': args.z0,
               'denominator_floor': args.denominator_floor,
               'valid_samples': int(valid.sum()), 'total_samples': int(len(valid)),
               'method': 'DC + complex cosine/sine least-squares fit at each specified frequency',
               'tone_fits': fits}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.output_dir / 'impedance.h5', 'x') as handle:
        handle.attrs['reference_impedance_ohm'] = args.z0
        handle.attrs['source_calibrated'] = str(args.calibrated)
        handle.create_dataset('time_s', data=time_s)
        handle.create_dataset('impedance_ohm', data=impedance)
        handle.create_dataset('valid_impedance', data=valid)
    (args.output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
