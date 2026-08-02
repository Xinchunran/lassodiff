#!/usr/bin/env python3
"""Entry point for staged mini_dev training; full runs require preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lassodiff.checkpoint_mini_v2 import ARCHITECTURE_ID, SCHEMA_VERSION
from lassodiff.training_mini_v2 import MiniTrainingSystemV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", choices=("backbone", "sidechain", "refiner", "joint"), default="backbone")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--preflight", required=True)
    args = parser.parse_args()
    preflight = json.loads(Path(args.preflight).read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("mini_dev V2 training requires a passing architecture preflight")
    system = MiniTrainingSystemV2(); system.configure_stage(args.stage)
    run = Path(args.run_dir)
    if run.exists() and any(run.iterdir()):
        raise RuntimeError("refusing to reuse a non-empty run directory")
    run.mkdir(parents=True, exist_ok=False)
    (run / "run_manifest.json").write_text(json.dumps({"architecture_id": ARCHITECTURE_ID, "schema_version": SCHEMA_VERSION, "stage": args.stage, "steps": args.steps}, indent=2) + "\n")
    raise RuntimeError("training data/optimizer wiring is intentionally gated; use the staged trainer after mini_v2 preflight")


if __name__ == "__main__":
    main()
