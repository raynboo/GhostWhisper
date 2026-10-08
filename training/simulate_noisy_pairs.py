#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostwhisper.audio import load_audio_mono
from ghostwhisper.simulation import (
    sample_real_calibrated_aggressive_noise_params,
    sample_real_calibrated_noise_params,
    simulate_noisy_waveform,
)
from ghostwhisper.visualization import save_pair_mel_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic noisy/clean waveform pairs from clean 48 kHz speech clips."
    )
    parser.add_argument(
        "--clean-manifest",
        type=Path,
        required=True,
        help="CSV manifest produced by prepare_clean_clips.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/simulated/vctk_48k_v1"),
        help="Directory where synthetic pairs and metadata will be written.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("artifacts/manifests/simulated_vctk_48k_v1.csv"),
        help="CSV manifest for generated noisy/clean pairs.",
    )
    parser.add_argument("--num-variants", type=int, default=1, help="Noisy variants per clean clip.")
    parser.add_argument("--max-clean-clips", type=int, default=None, help="Limit clean clips for quick tests.")
    parser.add_argument("--seed", type=int, default=20260426, help="Random seed.")
    parser.add_argument(
        "--buried-probability",
        type=float,
        default=0.9,
        help="Probability of generating a heavily noise-buried sample.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=10,
        help="Number of random generated pairs to visualize as mel figures.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("artifacts/previews/simulated_vctk_48k_v1"),
        help="Directory for preview mel comparison figures.",
    )
    parser.add_argument(
        "--simulation-profile",
        choices=["weakcarrier", "realcalib", "realcalib_aggressive"],
        default="weakcarrier",
        help="Synthetic EM noise profile.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = read_manifest(args.clean_manifest)
    if args.max_clean_clips is not None:
        rows = rows[: args.max_clean_clips]

    pair_rows: list[dict[str, str]] = []
    preview_candidates: list[dict[str, str]] = []
    noisy_root = args.output_dir / "noisy"
    meta_root = args.output_dir / "metadata"
    noisy_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    for row_index, row in enumerate(rows):
        clean_path = Path(row["audio_path"])
        if not clean_path.exists():
            continue
        clean, sample_rate = load_audio_mono(clean_path)
        for variant_index in range(args.num_variants):
            pair_id = f"{row['clip_id']}_v{variant_index:02d}"
            speaker_id = row.get("speaker_id", "unknown")
            noisy_dir = noisy_root / row.get("dataset", "dataset") / speaker_id
            meta_dir = meta_root / row.get("dataset", "dataset") / speaker_id
            noisy_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)

            params = None
            if args.simulation_profile == "realcalib":
                params = sample_real_calibrated_noise_params(rng, sample_rate)
            elif args.simulation_profile == "realcalib_aggressive":
                params = sample_real_calibrated_aggressive_noise_params(rng, sample_rate)
            noisy, params = simulate_noisy_waveform(
                clean,
                sample_rate,
                rng,
                params=params,
                buried_probability=args.buried_probability,
            )
            noisy_path = noisy_dir / f"{pair_id}.wav"
            metadata_path = meta_dir / f"{pair_id}.json"
            sf.write(noisy_path, noisy, sample_rate)

            metadata = {
                "pair_id": pair_id,
                "clean_manifest_row": row_index,
                "clean_audio_path": str(clean_path.resolve()),
                "noisy_audio_path": str(noisy_path.resolve()),
                "sample_rate": sample_rate,
                "duration_sec": len(clean) / sample_rate,
                "noise_params": params.to_dict(),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            pair_row = {
                "pair_id": pair_id,
                "clean_clip_id": row["clip_id"],
                "variant_index": str(variant_index),
                "clean_audio_path": str(clean_path.resolve()),
                "noisy_audio_path": str(noisy_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "sample_rate": str(sample_rate),
                "duration_sec": f"{len(clean) / sample_rate:.3f}",
                "snr_db": f"{params.snr_db:.3f}",
                "tone_count": str(params.tone_count),
                "transcript": row.get("transcript", ""),
            }
            pair_rows.append(pair_row)
            preview_candidates.append(pair_row)

    write_manifest(args.output_manifest, pair_rows)

    if args.preview_count > 0 and preview_candidates:
        preview_count = min(args.preview_count, len(preview_candidates))
        selected = rng.choice(len(preview_candidates), size=preview_count, replace=False)
        for preview_index, candidate_index in enumerate(selected):
            pair = preview_candidates[int(candidate_index)]
            clean, sample_rate = load_audio_mono(Path(pair["clean_audio_path"]))
            noisy, noisy_rate = load_audio_mono(Path(pair["noisy_audio_path"]))
            if noisy_rate != sample_rate:
                raise RuntimeError(f"Sample-rate mismatch in {pair['pair_id']}")
            save_pair_mel_figure(
                clean=clean,
                noisy=noisy,
                sample_rate=sample_rate,
                output_path=args.preview_dir / f"{preview_index:02d}_{pair['pair_id']}.png",
                title=f"{pair['pair_id']} | SNR {pair['snr_db']} dB | tones {pair['tone_count']}",
            )

    print(f"clean_manifest={args.clean_manifest}")
    print(f"num_clean_rows={len(rows)}")
    print(f"num_generated_pairs={len(pair_rows)}")
    print(f"output_dir={args.output_dir}")
    print(f"output_manifest={args.output_manifest}")
    print(f"preview_dir={args.preview_dir}")


if __name__ == "__main__":
    main()
