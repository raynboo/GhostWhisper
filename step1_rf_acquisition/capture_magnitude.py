#!/usr/bin/env python3
"""Capture one B210 IQ channel and save its magnitude trace as HDF5."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def display_path(path: Path) -> str:
    """Return an artifact-relative path without exposing a host filesystem path."""
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return path.name


class CaptureError(RuntimeError):
    """Raised when UHD is unavailable or the requested capture fails."""


def load_uhd() -> Any:
    try:
        import uhd
    except ImportError as exc:
        raise CaptureError(
            "UHD Python bindings are unavailable; install the system UHD package"
        ) from exc
    return uhd


def usrp_capture(
    center_freq_hz: float,
    sample_rate_hz: float,
    duration_s: float,
    gain_db: float,
    device_args: str,
) -> np.ndarray:
    """Capture on the B210-tested RX chain; reject incomplete/overflowed records."""
    uhd = load_uhd()
    usrp = uhd.usrp.MultiUSRP(device_args)
    num_samples = int(np.ceil(duration_s * sample_rate_hz))
    print(
        f"Capturing {num_samples:,} samples ({duration_s:g} s) at "
        f"{sample_rate_hz:g} S/s..."
    )
    usrp.set_rx_rate(sample_rate_hz, 0)
    actual_rate = float(usrp.get_rx_rate(0))
    if not np.isclose(actual_rate, sample_rate_hz, rtol=1e-6):
        raise CaptureError(f'Requested rate {sample_rate_hz} coerced to {actual_rate}; choose a supported rate')
    usrp.set_rx_freq(uhd.types.TuneRequest(center_freq_hz), 0)
    usrp.set_rx_gain(gain_db, 0)
    usrp.set_rx_antenna('RX2', 0)
    time.sleep(0.2)
    stream_args = uhd.usrp.StreamArgs('fc32', 'sc16')
    stream_args.channels = [0]
    streamer = usrp.get_rx_stream(stream_args)
    command = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    command.num_samps = num_samples
    command.stream_now = True
    samples = np.empty(num_samples, dtype=np.complex64)
    buffer = np.empty((1, streamer.get_max_num_samps()), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    received = 0
    deadline = time.monotonic() + duration_s + 5.0
    streamer.issue_stream_cmd(command)
    try:
        while received < num_samples and time.monotonic() < deadline:
            wanted = min(buffer.shape[1], num_samples - received)
            count = streamer.recv(buffer[:, :wanted], metadata, 1.0)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                raise CaptureError(f'Invalid capture (no file saved): {metadata.strerror()}')
            if count:
                samples[received:received + count] = buffer[0, :count]
                received += count
        if received != num_samples:
            raise CaptureError(f'Short capture: {received}/{num_samples}; no file saved')
        if not np.isfinite(samples).all():
            raise CaptureError('Non-finite IQ; no file saved')
        clipped = np.mean(np.maximum(abs(samples.real), abs(samples.imag)) >= 0.98)
        if clipped > 1e-4:
            raise CaptureError(f'RX clipping fraction {clipped}; lower gain and repeat')
    finally:
        stop = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        stop.stream_now = True
        streamer.issue_stream_cmd(stop)
    if samples.ndim != 1 or samples.size == 0:
        raise CaptureError(f"unexpected UHD result shape: {samples.shape}")
    print(f"Received {samples.size:,} samples.")
    return np.asarray(samples, dtype=np.complex64)


def calculate_magnitude(samples: np.ndarray) -> np.ndarray:
    """Return the magnitude time series for complex IQ samples."""
    return np.abs(samples).astype(np.float32, copy=False)


def save_plot(
    magnitude: np.ndarray,
    sample_rate_hz: float,
    path: Path,
    show: bool,
) -> None:
    """Save a time-domain magnitude plot."""
    time_axis = np.arange(magnitude.size, dtype=np.float64) / sample_rate_hz
    figure, axis = plt.subplots(figsize=(12, 6))
    stride = max(1, magnitude.size // 100000)
    axis.plot(time_axis[::stride], magnitude[::stride], linewidth=0.5)
    axis.set_title("Signal magnitude versus time")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Magnitude")
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def save_h5(
    magnitude: np.ndarray,
    sample_rate_hz: float,
    center_freq_hz: float,
    duration_s: float,
    path: Path,
) -> None:
    """Save magnitude, time, and non-identifying capture parameters to HDF5."""
    time_axis = np.arange(magnitude.size, dtype=np.float64) / sample_rate_hz
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "magnitude", data=magnitude, compression="gzip", compression_opts=4
        )
        handle.create_dataset(
            "time", data=time_axis, compression="gzip", compression_opts=4
        )
        handle.attrs["sample_rate_hz"] = sample_rate_hz
        handle.attrs["center_frequency_hz"] = center_freq_hz
        handle.attrs["requested_duration_s"] = duration_s
        handle.attrs["num_samples"] = magnitude.size
        handle.attrs["capture_time_utc"] = datetime.now(timezone.utc).isoformat()


def save_csv(magnitude: np.ndarray, sample_rate_hz: float, path: Path) -> None:
    """Save the magnitude trace as a two-column CSV file."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "magnitude"])
        writer.writerows(
            (index / sample_rate_hz, float(value))
            for index, value in enumerate(magnitude)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one B210 channel and save a magnitude trace."
    )
    parser.add_argument("--center-freq", type=float, default=601e6, metavar="HZ")
    parser.add_argument("--sample-rate", type=float, default=5e6, metavar="SPS")
    parser.add_argument("--duration", type=float, default=28.0, metavar="SECONDS")
    parser.add_argument("--gain", type=float, default=10.0, metavar="DB")
    parser.add_argument(
        "--serial",
        default="",
        help="optional B210 serial; omit to use the first matching B200-series device",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "captures")
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true", help="display the plot interactively")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    values = (args.center_freq, args.sample_rate, args.duration, args.gain)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("all numeric parameters must be finite")
    if args.center_freq <= 0 or args.sample_rate <= 0 or args.duration <= 0:
        raise ValueError("frequency, sample rate, and duration must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        device_args = "type=b200"
        if args.serial.strip():
            device_args += f",serial={args.serial.strip()}"
        samples = usrp_capture(
            args.center_freq,
            args.sample_rate,
            args.duration,
            args.gain,
            device_args,
        )
        magnitude = calculate_magnitude(samples)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"magnitude_{timestamp}"
        h5_path = args.output_dir / f"{stem}.h5"
        save_h5(magnitude, args.sample_rate, args.center_freq, args.duration, h5_path)
        print(f"HDF5: {display_path(h5_path)}")
        if not args.no_plot:
            plot_path = args.output_dir / f"{stem}.png"
            save_plot(magnitude, args.sample_rate, plot_path, args.show)
            print(f"Plot: {display_path(plot_path)}")
        if args.save_csv:
            csv_path = args.output_dir / f"{stem}.csv"
            save_csv(magnitude, args.sample_rate, csv_path)
            print(f"CSV: {display_path(csv_path)}")
        return 0
    except (CaptureError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
