#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lassodiff.data.family_split import validate_family_split_manifest
from lassodiff.data.lassopred_lmdb import LassoPredLMDBDataset
from lassodiff.threading_truth import (
    build_threading_alignment_report, validate_threading_alignment_report,
    write_threading_alignment_report,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    validate_family_split_manifest(split)
    dataset = LassoPredLMDBDataset(args.dataset)
    report = build_threading_alignment_report(
        dataset, split_manifest_sha256=split.get("manifest_sha256", ""),
    )
    validate_threading_alignment_report(report)
    write_threading_alignment_report(report, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
