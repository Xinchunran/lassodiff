#!/usr/bin/env python3
"""Build the validated LassoPred JSON--PDB LMDB dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lassodiff.data.lassopred_lmdb import build_lassopred_lmdb, make_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--structure-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reject-report", default=None)
    parser.add_argument("--map-size-gb", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    manifest = build_lassopred_lmdb(
        args.metadata_json, args.structure_dir, args.output, args.reject_report,
        map_size=args.map_size_gb * 1024**3, overwrite=args.overwrite, verify_only=args.verify_only,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not args.verify_only:
        split = make_split_manifest(manifest["record_ids"], args.split_seed, args.val_fraction, args.test_fraction)
        path = Path(args.output) / "split.json"
        path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote split manifest: {path}")


if __name__ == "__main__":
    main()
