#!/usr/bin/env python3
"""Fail-closed, device-independent startup gate for the V3 structure pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lassodiff.architecture_contract_v3 import load_architecture_config_v3
from lassodiff.data.family_split import validate_family_split_manifest
from lassodiff.threading_truth import validate_threading_alignment_report


def verify_startup(config_path, split_path, preflight_path, cache_manifest_path, threading_report_path):
    config = load_architecture_config_v3(config_path)
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    validate_family_split_manifest(split)
    preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("trace", {}).get("reasoner_called") != 1:
        raise RuntimeError("preflight is not a PASS with exactly one reasoner call")
    state = preflight.get("reasoning_state", {})
    for key, expected in (
        ("checkpoint_sha256", config.reasoning.checkpoint_sha256),
        ("opendde_commit", config.reasoning.opendde_commit),
        ("feature_schema_version", config.reasoning.cache_schema_version),
    ):
        if state.get(key) != expected:
            raise RuntimeError(f"preflight provenance mismatch at {key}")
    cache = json.loads(Path(cache_manifest_path).read_text(encoding="utf-8"))
    expected_cache = {
        "status": "PASS", "sequence_count": split["unique_sequence_count"],
        "checkpoint_sha256": config.reasoning.checkpoint_sha256,
        "opendde_commit": config.reasoning.opendde_commit,
        "n_cycle": config.reasoning.n_cycle,
        "feature_schema_version": config.reasoning.cache_schema_version,
    }
    if any(cache.get(key) != value for key, value in expected_cache.items()):
        raise RuntimeError("reasoning cache manifest does not match config/split")
    threading = json.loads(Path(threading_report_path).read_text(encoding="utf-8"))
    validate_threading_alignment_report(threading)
    if threading.get("split_manifest_sha256") != split.get("manifest_sha256"):
        raise RuntimeError("threading report split hash does not match locked split")
    if config.model.diffusion_blocks != preflight.get("trace", {}).get("geometry_calls"):
        raise RuntimeError("preflight diffusion block count does not match config")
    return {
        "status": "PASS", "architecture_id": config.architecture_id,
        "split_manifest_sha256": split["manifest_sha256"],
        "threading_alignment_report_sha256": threading["report_sha256"],
        "train_records": len(split["train"]), "validation_records": len(split["val"]),
        "test_records": len(split["test"]), "global_batch_size": 64,
        "world_size": 4, "batch_size_per_gpu": 16,
        "reasoning_checkpoint_sha256": config.reasoning.checkpoint_sha256,
        "threading_checker_version": threading["checker_version"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--threading-report", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    try:
        result = verify_startup(
            args.config, args.split, args.preflight, args.cache_manifest, args.threading_report,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
