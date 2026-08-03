#!/usr/bin/env python3
"""Build strict decoder-consistent Mini V2 target labels offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.data.mini_grouped_pdb_dataset import GroupedMiniPDBDataset
from lassodiff.seq_encoder import seq_to_aa_ids
from lassodiff.target_fit_mini_v2 import (
    DECODER_FIT_VERSION,
    decoder_fit_cache_key,
    fit_decoder_consistent_target,
    save_decoder_fit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--structure-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record-ids", nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = GroupedMiniPDBDataset(
        args.metadata, args.structure_root, record_ids=args.record_ids,
    )
    rows = []
    for item in dataset.examples:
        candidate = CandidateCondition(item["sequence"], int(item["k"]), int(item["p"]))
        aa_ids = seq_to_aa_ids(candidate.sequence)
        for conformer in torch.where(item["conformer_mask"])[0].tolist():
            source = item["conformer_sources"][conformer]
            key = decoder_fit_cache_key(source, candidate)
            result = fit_decoder_consistent_target(
                sequence=candidate.sequence,
                candidate=candidate,
                canonical_core=item["core_targets"][conformer],
                core_mask=item["core_target_masks"][conformer],
                atom14_target=item["atom14_targets"][conformer],
                atom14_mask=item["atom14_target_masks"][conformer],
                aa_ids=aa_ids,
                backbone_torsions=item["backbone_torsions"][conformer],
                backbone_mask=item["backbone_torsion_masks"][conformer],
                chi_angles=item["chi_targets"][conformer],
                chi_mask=item["chi_masks"][conformer],
                max_steps=args.max_steps,
            )
            save_decoder_fit(output / f"{key}.pt", result, source=source,
                             candidate=candidate, cache_key=key)
            rows.append({
                "record_id": item["record_id"], "rank": item["conformer_ranks"][conformer],
                "k": candidate.k, "p": candidate.p, "source": source, "cache_key": key,
                "converged": result.converged, "strict_valid": result.strict_valid,
                "canonical_ca_rmsd": result.canonical_ca_rmsd, "lddt": result.lddt,
                "rejection_reasons": list(result.strict_rejection_reasons),
            })
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    payload = {"fit_version": DECODER_FIT_VERSION, "targets": rows}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
