"""Measure real Mini V2 remediation metrics without training or templates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.chi_geometry import build_atom14_from_rigid_groups
from lassodiff.data.mini_grouped_pdb_dataset import GroupedMiniPDBDataset
from lassodiff.lasso_core_decoder import decode_lasso_core
from lassodiff.metrics_mini_v2 import aligned_ca_rmsd, lddt_score
from lassodiff.seq_encoder import seq_to_aa_ids
from lassodiff.torsion_state import TorsionState
from lassodiff.target_fit_mini_v2 import DECODER_FIT_VERSION
from lassodiff.validation.strict_lasso import strict_lasso_check


def measure(metadata: str, structure_root: str, record_id: str | None = None,
            target_fit_cache: str | None = None) -> dict:
    dataset = GroupedMiniPDBDataset(metadata, structure_root,
                                    record_ids=[record_id] if record_id else None,
                                    decoder_fit_cache=target_fit_cache,
                                    require_decoder_fit=target_fit_cache is not None)
    rows = []
    for item in dataset.examples:
        for conformer, active in enumerate(item["conformer_mask"].tolist()):
            if not active:
                continue
            target = item["core_targets"][conformer]
            torsions = item["backbone_torsions"][conformer]
            candidate = CandidateCondition(item["sequence"], int(item["k"]), int(item["p"]))
            state = TorsionState(
                torsions[None, None], item["backbone_torsion_masks"][conformer][None, None],
                item["chi_targets"][conformer][None, None], item["chi_masks"][conformer][None, None],
            )
            reconstructed = decode_lasso_core(
                state, sequences=[candidate.sequence], candidates=[candidate],
                token_mask=torch.ones((1, len(candidate.sequence)), dtype=torch.bool),
            )[0, 0]
            atom14, atom14_mask = build_atom14_from_rigid_groups(
                reconstructed, seq_to_aa_ids(candidate.sequence),
                item["chi_targets"][conformer], item["chi_masks"][conformer], candidate,
            )
            mask = item["core_target_masks"][conformer]
            ca_mask = mask[:, 1]
            ca_rmsd = torch.sqrt(
                ((reconstructed[:, 1] - target[:, 1]).square().sum(-1)[ca_mask]).mean()
            )
            lddt = lddt_score(reconstructed[ca_mask, 1], target[ca_mask, 1])
            core_strict = strict_lasso_check(
                reconstructed, item["core_target_masks"][conformer], candidate,
            )
            full_strict = strict_lasso_check(
                reconstructed, item["core_target_masks"][conformer], candidate,
                atom14_coordinates=atom14, atom14_atom_mask=atom14_mask,
            )
            fit_metrics = item.get("decoder_fit_metrics", ())
            fit_metrics = fit_metrics[conformer] if conformer < len(fit_metrics) else {}
            rows.append({
                "record_id": item["record_id"],
                "rank": int(item.get("conformer_ranks", (conformer + 1,))[conformer]),
                "sequence": item["sequence"],
                "k": int(item["k"]),
                "p": int(item["p"]),
                "loop_size": int(item.get("loop_size", int(item["k"]) + 1)),
                "target_source": item.get("conformer_sources", ("",))[conformer],
                "finite": bool(torch.isfinite(reconstructed).all()),
                "canonical_ca_rmsd": float(ca_rmsd),
                "aligned_ca_rmsd": float(aligned_ca_rmsd(reconstructed[ca_mask, 1], target[ca_mask, 1])),
                "true_lddt": float(lddt),
                "decoder_fit_to_raw_ca_rmsd": fit_metrics.get("ca_rmsd"),
                "decoder_fit_to_raw_lddt": fit_metrics.get("lddt"),
                "raw_target_strict_valid": fit_metrics.get("raw_target_strict_valid"),
                "core_strict_valid": bool(core_strict.valid),
                "core_strict_rejection_reasons": list(core_strict.rejection_reasons),
                "full_atom_strict_valid": bool(full_strict.valid),
                "full_atom_strict_rejection_reasons": list(full_strict.rejection_reasons),
                # Backwards-compatible aliases always mean the final full-atom gate.
                "strict_valid": bool(full_strict.valid),
                "strict_rejection_reasons": list(full_strict.rejection_reasons),
                "backbone_valid": bool(full_strict.backbone_valid),
                "formed_geometry": bool(full_strict.formed_geometry_valid),
                "crossing_count": int(full_strict.crossing_count),
                "clash_valid": bool(full_strict.clash_valid),
            })
    gate_ok = bool(rows) and all(
        row["finite"] and row["full_atom_strict_valid"]
        and row["canonical_ca_rmsd"] < 0.75 and row["true_lddt"] > 0.85
        for row in rows
    )
    try:
        source_commit = subprocess.check_output(
            ["git", "--git-dir=.git-mini-dev", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unknown"
    return {
        "status": "PASS" if gate_ok else "FAIL",
        "architecture_id": "lassodiff_mini_torsion_v2",
        "schema_version": 2,
        "source_commit": source_commit,
        "canonicalization_version": "mini_root_frame_v1",
        "decoder_fit_version": DECODER_FIT_VERSION if target_fit_cache else None,
        "template_coordinates_used": False,
        "prior_mode": "extract_decode_roundtrip",
        "dataset_mapping_sha256": dataset.mapping_sha256,
        "strict_checker": "lassodiff.validation.strict_lasso",
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--structure-root", required=True)
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--target-fit-cache", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = measure(args.metadata, args.structure_root, args.record_id, args.target_fit_cache)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in report["samples"]),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
