#!/usr/bin/env python3
"""Build a Mini 5-fold manifest while preserving the locked V3 test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lassodiff.data.family_split import validate_family_split_manifest
from lassodiff.data.mini_dataset import MiniLassoDataset
from lassodiff.data.mini_split import build_mini_cv_manifest, validate_mini_cv_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--structure-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    source = json.loads(Path(args.source_split).read_text(encoding="utf-8"))
    validate_family_split_manifest(source)
    source_ids = set(source["train"]) | set(source["val"]) | set(source["test"])
    dataset = MiniLassoDataset(args.metadata, args.structure_root, record_ids=source_ids)
    qualified = {example["record_id"] for example in dataset.examples}
    unavailable = sorted(source_ids - qualified)
    manifest = build_mini_cv_manifest(
        source, unavailable_record_ids=unavailable, fold_count=5, seed=args.seed,
    )
    validate_mini_cv_manifest(manifest, source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "manifest_sha256": manifest["manifest_sha256"],
        "development_records": manifest["development_record_count"],
        "locked_test_records": manifest["locked_test_record_count"],
        "unavailable_records": unavailable,
        "qualified_candidate_examples": len(dataset),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
