#!/usr/bin/env python3
"""Run the paper-sized two-model pipeline without hardware or transcripts."""
from pathlib import Path
import argparse
import json
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ghostwhisper.io_utils import read_audio, write_audio, read_manifest, sha256
from ghostwhisper.restoration import ResUNetRestorer, RefinerRestorer
from ghostwhisper.stable_tone_filter import filter_stable_tones


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="CSV listing user-supplied input WAV files")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/demo")
    p.add_argument("--denoiser", type=Path, required=True, help="User-supplied denoiser checkpoint")
    p.add_argument("--refiner", type=Path, required=True, help="User-supplied refiner checkpoint")
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--skip-notch", action="store_true", help="Ablation only; recorded in run.json")
    p.add_argument("--allow-existing", action="store_true", help="Explicitly allow replacing prior outputs")
    args = p.parse_args()
    for label in ("manifest", "denoiser", "refiner"):
        if not getattr(args, label).is_file():
            p.error(f"--{label}: file not found: {getattr(args, label)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing:
        p.error("Output directory is not empty; choose a new directory or --allow-existing")
    if args.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)
    name = args.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    defaults = SimpleNamespace(sample_rate=48000, resunet_chunk_seconds=4.0,
                               refiner_clip_seconds=4.0, resunet_base_channels=64)
    denoiser = ResUNetRestorer(args.denoiser, defaults, device)
    refiner = RefinerRestorer(args.refiner, defaults, device)
    counts = {"denoiser": sum(x.numel() for x in denoiser.model.parameters()),
              "refiner": sum(x.numel() for x in refiner.model.parameters())}
    if counts != {"denoiser": 26298817, "refiner": 17411522}:
        p.error(f"Checkpoint architecture differs from the paper-sized artifact: {counts}")
    if denoiser.sample_rate != 48000 or refiner.sample_rate != 48000:
        p.error("This artifact requires 48 kHz checkpoints")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)
    records = []
    for row in rows:
        t0 = time.perf_counter()
        folder = args.output_dir / row["sample_id"]
        raw = read_audio(row["raw_audio"])
        pre, tones = (raw, []) if args.skip_notch else filter_stable_tones(raw, 48000)
        denoised = denoiser.restore(pre)
        enhanced = refiner.restore(denoised)
        for stage, x in [("raw", raw), ("preprocessed", pre), ("denoised", denoised), ("restored", enhanced)]:
            write_audio(folder / f"{stage}.wav", x)
        record = {"sample_id": row["sample_id"], "input_sha256": sha256(row["raw_audio"]),
                  "duration_seconds": len(raw) / 48000,
                  "elapsed_seconds": time.perf_counter() - t0, "notches": [asdict(t) for t in tones]}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {"pipeline": "notch -> log-Mel ResUNet -> STFT residual refiner",
              "device": name, "torch": torch.__version__, "parameters": counts,
              "skip_notch": args.skip_notch, "denoiser_sha256": sha256(args.denoiser),
              "refiner_sha256": sha256(args.refiner), "samples": records}
    (args.output_dir / "run.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
