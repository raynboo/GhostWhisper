#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostwhisper.audio import load_audio_mono, resample_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample clean speech to 48 kHz and chop it into fixed-duration clips."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Input CSV manifest.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for clip WAVs.")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="CSV manifest for the generated fixed-length clips.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=48000,
        help="Target sample rate for the clean clips.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=4.0,
        help="Length of each exported clip in seconds.",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=1.5,
        help="Discard source utterances shorter than this duration.",
    )
    return parser.parse_args()


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def iter_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_id",
                "dataset",
                "speaker_id",
                "utterance_id",
                "clip_index",
                "audio_path",
                "transcript",
                "sample_rate",
                "duration_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = iter_manifest_rows(args.manifest)
    clip_rows: list[dict[str, str]] = []
    clip_length = int(args.clip_seconds * args.target_sample_rate)

    for row in rows:
        source_path = Path(row["audio_path"])
        if not source_path.exists():
            continue
        samples, source_rate = load_audio_mono(source_path)
        if len(samples) / source_rate < args.min_seconds:
            continue
        samples = resample_audio(samples, source_rate, args.target_sample_rate)
        usable = len(samples) // clip_length
        if usable == 0:
            continue

        for clip_index in range(usable):
            start = clip_index * clip_length
            end = start + clip_length
            clip = samples[start:end]
            clip_id = stable_id(f"{row['dataset']}::{row['utterance_id']}::{clip_index}")
            speaker_dir = args.output_dir / row["dataset"] / row["speaker_id"]
            speaker_dir.mkdir(parents=True, exist_ok=True)
            clip_path = speaker_dir / f"{row['utterance_id']}_{clip_index:03d}_{clip_id}.wav"
            sf.write(clip_path, clip, args.target_sample_rate)
            clip_rows.append(
                {
                    "clip_id": clip_id,
                    "dataset": row["dataset"],
                    "speaker_id": row["speaker_id"],
                    "utterance_id": row["utterance_id"],
                    "clip_index": str(clip_index),
                    "audio_path": str(clip_path.resolve()),
                    "transcript": row.get("transcript", ""),
                    "sample_rate": str(args.target_sample_rate),
                    "duration_sec": f"{args.clip_seconds:.3f}",
                }
            )

    write_rows(args.output_manifest, clip_rows)
    print(f"input_manifest={args.manifest}")
    print(f"num_input_rows={len(rows)}")
    print(f"num_output_clips={len(clip_rows)}")
    print(f"output_dir={args.output_dir}")
    print(f"output_manifest={args.output_manifest}")


if __name__ == "__main__":
    main()
