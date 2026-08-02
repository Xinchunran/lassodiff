#!/usr/bin/env python3
"""Data-parallel, sequence-keyed OpenDDE residue reasoning cache builder."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lassodiff.architecture_contract_v3 import load_architecture_config_v3
from lassodiff.data.lassopred_lmdb import LassoPredLMDBDataset
from lassodiff.opendde_bridge.cache import OpenDDEReasoningCache, ReasoningCacheIdentity
from lassodiff.opendde_bridge.feature_builder import OpenDDEFeatureBuilder
from lassodiff.opendde_bridge.loader import PinnedOpenDDEConfig, load_pinned_opendde_reasoner


def _append(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _unique_sequences(dataset):
    # Read the validated LMDB records directly to avoid materialising every MD
    # coordinate target merely to obtain candidate-independent sequence text.
    from lassodiff.data.lassopred_lmdb import _RECORD_PREFIX
    sequences = set()
    env = dataset._open()
    with env.begin() as txn:
        for record_id in dataset.record_ids:
            payload = txn.get(_RECORD_PREFIX + record_id.encode("utf-8"))
            if payload is None:
                raise RuntimeError(f"LMDB record disappeared: {record_id}")
            sequences.add(str(pickle.loads(payload)["sequence"]))
    return sorted(sequences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset")
    source.add_argument("--sequence-manifest")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_architecture_config_v3(args.config)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("OpenDDE cache requested CUDA but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = "cuda"
    if args.dataset:
        dataset = LassoPredLMDBDataset(args.dataset)
        sequences = _unique_sequences(dataset)
    else:
        from lassodiff.data.sequence_v3 import load_sequence_manifest
        _payload, examples = load_sequence_manifest(args.sequence_manifest)
        sequences = sorted({example.sequence for example in examples})
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        sequences = sequences[:args.limit]
    shard = sequences[rank::world_size]
    log_path = Path(args.log_dir) / f"cache.rank{rank:04d}.jsonl"
    cache = OpenDDEReasoningCache(config.reasoning.cache_dir)

    def identity(sequence):
        return ReasoningCacheIdentity(
            sequence=sequence, opendde_commit=config.reasoning.opendde_commit,
            checkpoint_sha256=config.reasoning.checkpoint_sha256,
            feature_schema_version=config.reasoning.cache_schema_version,
            use_msa=config.reasoning.use_msa, use_template=config.reasoning.use_template,
            dtype=config.reasoning.dtype, n_cycle=config.reasoning.n_cycle,
        )

    pending = []
    for sequence in shard:
        item = identity(sequence)
        try:
            cache.get(item)
            if args.overwrite:
                pending.append((sequence, item))
        except RuntimeError:
            pending.append((sequence, item))
    _append(log_path, {
        "event": "start", "rank": rank, "world_size": world_size,
        "unique_sequence_count": len(sequences), "shard_count": len(shard), "pending_count": len(pending),
        "checkpoint_sha256": config.reasoning.checkpoint_sha256,
        "opendde_commit": config.reasoning.opendde_commit, "n_cycle": config.reasoning.n_cycle,
    })
    if not pending:
        _append(log_path, {"event": "complete", "rank": rank, "written": 0, "skipped": len(shard)})
        return
    reasoner, load_manifest = load_pinned_opendde_reasoner(PinnedOpenDDEConfig(
        source_root=config.reasoning.source_root, runtime_root=config.reasoning.runtime_root,
        checkpoint_path=config.reasoning.checkpoint_path,
        checkpoint_sha256=config.reasoning.checkpoint_sha256,
        opendde_commit=config.reasoning.opendde_commit,
        expected_numel=config.reasoning.expected_numel, n_cycle=config.reasoning.n_cycle,
        device=device, dtype=config.reasoning.dtype,
    ))
    builder = OpenDDEFeatureBuilder(config.reasoning.runtime_root)
    started = time.time()
    for index, (sequence, item) in enumerate(pending, start=1):
        state = reasoner(builder.build(sequence, name=f"cache_rank{rank}_{index}"))
        destination = cache.put(item, state, overwrite=args.overwrite)
        _append(log_path, {
            "event": "cached", "rank": rank, "index": index, "pending_count": len(pending),
            "sequence_sha256": item.sequence_sha256, "length": len(sequence),
            "single_shape": list(state.single.shape), "pair_shape": list(state.pair.shape),
            "path": str(destination), "elapsed_seconds": time.time() - started,
        })
    _append(log_path, {
        "event": "complete", "rank": rank, "written": len(pending), "skipped": len(shard) - len(pending),
        "elapsed_seconds": time.time() - started, "load_manifest": load_manifest.__dict__,
    })


if __name__ == "__main__":
    main()
