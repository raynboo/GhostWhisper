#!/usr/bin/env python3
"""Export a trusted historical checkpoint to a portable weights-only bundle."""
import argparse
import json
import hashlib
from pathlib import Path
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--trusted-legacy", action="store_true", help="Permit pickle ONLY for a trusted local training checkpoint")
    a = p.parse_args()
    if a.destination.exists():
        p.error("Destination already exists")
    c = torch.load(a.source, map_location="cpu", weights_only=not a.trusted_legacy)
    config = {k: v for k, v in c.get("config", {}).items()
              if k not in {"manifest", "output_dir", "resume_checkpoint", "device", "device_resolved"}}
    config = json.loads(json.dumps(config, default=str))
    a.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": c["model"], "config": config, "epoch": c.get("epoch")}, a.destination)
    report = {"source_name": a.source.name, "source_sha256": digest(a.source),
              "bundle_sha256": digest(a.destination), "epoch": c.get("epoch"),
              "state_elements": sum(x.numel() for x in c["model"].values()),
              "training_config": c.get("config", {})}
    a.destination.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, default=str))


if __name__ == "__main__":
    main()
