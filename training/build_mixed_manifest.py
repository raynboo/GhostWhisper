#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synthetic+real mixed manifest for fine-tuning.")
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-count", type=int, default=12000)
    parser.add_argument("--real-repeat", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--path-prefix-from", type=str, default="")
    parser.add_argument("--path-prefix-to", type=str, default="")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rewrite_paths(row: dict[str, str], prefix_from: str, prefix_to: str) -> dict[str, str]:
    if not prefix_from:
        return row
    updated = dict(row)
    for key, value in row.items():
        if value.startswith(prefix_from):
            updated[key] = prefix_to + value[len(prefix_from) :]
    return updated


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    synthetic_rows = read_rows(args.synthetic_manifest)
    real_rows = read_rows(args.real_manifest)
    if args.synthetic_count < len(synthetic_rows):
        selected = rng.choice(len(synthetic_rows), size=args.synthetic_count, replace=False)
        synthetic_rows = [synthetic_rows[int(idx)] for idx in selected]
    real_rows = [rewrite_paths(row, args.path_prefix_from, args.path_prefix_to) for row in real_rows]

    mixed_rows: list[dict[str, str]] = []
    mixed_rows.extend(synthetic_rows)
    for repeat_idx in range(args.real_repeat):
        for row in real_rows:
            repeated = dict(row)
            repeated["pair_id"] = f"{row['pair_id']}_r{repeat_idx:03d}"
            mixed_rows.append(repeated)
    rng.shuffle(mixed_rows)

    fieldnames = sorted(set().union(*(row.keys() for row in mixed_rows)))
    preferred = [
        "pair_id",
        "clean_clip_id",
        "variant_index",
        "clean_audio_path",
        "noisy_audio_path",
        "metadata_path",
        "sample_rate",
        "duration_sec",
        "snr_db",
        "tone_count",
        "transcript",
        "source",
    ]
    fieldnames = [field for field in preferred if field in fieldnames] + [
        field for field in fieldnames if field not in preferred
    ]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mixed_rows)

    print(f"synthetic_rows={len(synthetic_rows)}")
    print(f"real_rows={len(real_rows)}")
    print(f"real_repeat={args.real_repeat}")
    print(f"mixed_rows={len(mixed_rows)}")
    print(f"output_manifest={args.output_manifest}")


if __name__ == "__main__":
    main()
