#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostwhisper.dataset_manifest import scan_dataset, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean-speech manifest from a public speech dataset."
    )
    parser.add_argument(
        "--dataset-type",
        choices=["vctk", "librispeech", "common_voice"],
        default="vctk",
        help="Dataset layout parser to use. Defaults to VCTK for the local setup.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local path of a user-supplied speech dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV file to write the manifest to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = scan_dataset(args.dataset_root, args.dataset_type)
    write_manifest(entries, args.output)
    print(f"dataset_type={args.dataset_type}")
    print(f"dataset_root={args.dataset_root}")
    print(f"num_entries={len(entries)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
