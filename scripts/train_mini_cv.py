#!/usr/bin/env python3
"""Run all five strict Mini CV folds sequentially on one four-GPU node."""
from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from lassodiff.data.mini_split import validate_mini_cv_manifest


def _write(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--structure-root", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if sorted(set(args.folds)) != sorted(args.folds) or any(not 0 <= fold < 5 for fold in args.folds):
        raise ValueError("folds must be unique values in [0,4]")
    source_split_path = Path(args.source_split).resolve()
    split_path = Path(args.split).resolve()
    source = json.loads(source_split_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_mini_cv_manifest(split, source)
    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    expected = (split["manifest_sha256"], source["manifest_sha256"], 5)
    observed = (
        preflight.get("cv_split_manifest_sha256"),
        preflight.get("source_split_manifest_sha256"),
        preflight.get("cv_fold_count"),
    )
    if preflight.get("status") != "PASS" or observed != expected:
        raise RuntimeError("Mini CV launcher requires a matching PASS preflight")
    run_root = Path(args.run_root)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty CV run root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True,
    ).strip()
    manifest = {
        "status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": revision, "folds": args.folds, "completed_folds": [],
        "active_fold": None, "steps_per_fold": args.steps,
        "cv_split_manifest_sha256": split["manifest_sha256"],
        "source_split_manifest_sha256": source["manifest_sha256"],
        "locked_test_record_count": split["locked_test_record_count"],
        "locked_test_loaded_during_training": False,
    }
    status_path = run_root / "cv_status.json"
    _write(status_path, manifest)
    completed = False

    def mark_failed():
        if not completed:
            failed = dict(manifest)
            failed["status"] = "FAILED"
            failed["reason"] = "launcher_exit_without_completion"
            _write(status_path, failed)

    atexit.register(mark_failed)
    torchrun = Path(sys.executable).with_name("torchrun")
    if not torchrun.is_file():
        raise FileNotFoundError(f"torchrun is missing next to the active Python: {torchrun}")
    common = [
        str(torchrun), "--standalone", "--nproc-per-node=4", "-m", "scripts.train_mini",
        "--metadata", str(Path(args.metadata).resolve()),
        "--structure-root", str(Path(args.structure_root).resolve()),
        "--preflight", str(Path(args.preflight).resolve()),
        "--split", str(split_path), "--source-split", str(source_split_path),
        "--steps", str(args.steps), "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers), "--log-every", str(args.log_every),
        "--save-every", str(args.save_every), "--validate-every", str(args.validate_every),
        "--validation-max-batches", str(args.validation_max_batches), "--seed", str(args.seed),
    ]
    for fold in args.folds:
        manifest["active_fold"] = fold
        _write(status_path, manifest)
        command = common + [
            "--fold", str(fold), "--run-dir", str((run_root / f"fold-{fold}").resolve()),
        ]
        subprocess.run(command, check=True)
        manifest["completed_folds"].append(fold)
        manifest["active_fold"] = None
        _write(status_path, manifest)
    manifest["status"] = "COMPLETE"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write(status_path, manifest)
    completed = True


if __name__ == "__main__":
    main()
