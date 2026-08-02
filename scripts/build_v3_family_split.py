#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lassodiff.data.family_split import build_family_split
from lassodiff.data.lassopred_lmdb import LassoPredLMDBDataset, _RECORD_PREFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--max-sequence-distance", type=float, default=.4)
    parser.add_argument("--min-stratum-size", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=.1)
    parser.add_argument("--test-fraction", type=float, default=.1)
    args = parser.parse_args()
    dataset = LassoPredLMDBDataset(args.dataset)
    records, topologies = {}, {}
    with dataset._open().begin() as txn:
        for record_id in dataset.record_ids:
            row = pickle.loads(txn.get(_RECORD_PREFIX + record_id.encode()))
            records[record_id] = row["sequence"]
            topologies[record_id] = row["candidates"]
    manifest = build_family_split(
        records, topologies, seed=args.seed, val_fraction=args.val_fraction,
        test_fraction=args.test_fraction, max_sequence_distance=args.max_sequence_distance,
        min_stratum_size=args.min_stratum_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "record_count", "unique_sequence_count", "cluster_count", "compared_pair_count", "manifest_sha256",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
