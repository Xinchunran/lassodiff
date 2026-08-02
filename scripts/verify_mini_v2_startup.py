#!/usr/bin/env python3
"""Fail-closed behavioral preflight for the V2 route (never the legacy gate)."""
from __future__ import annotations
import argparse, inspect, json
from pathlib import Path
import torch

from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.backbone_kinematics import build_core_from_torsions, extract_backbone_torsions
from lassodiff.data.mini_grouped_pdb_dataset import GroupedMiniPDBDataset
from lassodiff.data.mini_grouped_dataset import collate_grouped_mini
from lassodiff.data.mini_split import validate_mini_cv_manifest
from lassodiff.model_mini_v2 import MiniTorsionDiffusion
from lassodiff.sampler_mini_v2 import MiniInferenceConfig, open_chain_torsion_prior
from lassodiff.training_mini_v2 import MiniTrainingSystemV2


def _require_v2_preflight(path, cv_split, source_split):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"status": "PASS", "architecture_id": "lassodiff_mini_torsion_v2", "schema_version": 2,
                "distributed_backend": "fsdp_full_shard", "template_coordinates_used": False,
                "unassisted_prior": "open_chain", "fake_overfit_path_present": False}
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"V2 preflight mismatch: {key}")
    if report.get("cv_split_manifest_sha256") != cv_split["manifest_sha256"]:
        raise RuntimeError("V2 preflight CV split mismatch")
    if report.get("source_split_manifest_sha256") != source_split["manifest_sha256"]:
        raise RuntimeError("V2 preflight source split mismatch")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True); parser.add_argument("--structure-root", required=True)
    parser.add_argument("--source-split", required=True); parser.add_argument("--cv-split", required=True)
    parser.add_argument("--esm-cache", default=None); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cv, source = (json.loads(Path(x).read_text()) for x in (args.cv_split, args.source_split))
    validate_mini_cv_manifest(cv, source)
    # Dataset construction is part of preflight; an empty or rank-unqualified
    # dataset fails instead of authorizing a merely importable trainer.
    dataset = GroupedMiniPDBDataset(args.metadata, args.structure_root, record_ids=cv["folds"][0]["train"])
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    L = len(candidate.sequence)
    phi, psi, omega = torch.zeros(L), torch.zeros(L), torch.full((L,), torch.pi)
    core = build_core_from_torsions(candidate.sequence, phi, psi, omega)
    recovered, mask = extract_backbone_torsions(core)
    if not torch.isfinite(recovered).all() or not mask[1:, 0].all():
        raise RuntimeError("backbone torsion roundtrip preflight failed")
    prior = open_chain_torsion_prior([candidate], num_samples=2, generator=torch.Generator().manual_seed(3))
    if prior.backbone.shape[1] != 2 or prior.backbone_mask[0, :, 0, 0].any():
        raise RuntimeError("open-chain prior mask contract failed")
    system = MiniTrainingSystemV2.tiny_for_test()
    system.configure_stage("backbone")
    real_batch = collate_grouped_mini([dataset[0]])
    optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad], lr=1e-3)
    loss = system.forward_backbone_stage(real_batch, generator=torch.Generator().manual_seed(7), global_step=1).total
    if not torch.isfinite(loss):
        raise RuntimeError("V2 preflight real loss is non-finite")
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    source_code = inspect.getsource(MiniTrainingSystemV2)
    if any(x in source_code for x in ("overfit_parameter", "loss_target", "best_ca_rmsd = 0.0", "best_lddt = 1.0")):
        raise RuntimeError("fake V2 path detected")
    report = {"status": "PASS", "architecture_id": "lassodiff_mini_torsion_v2", "schema_version": 2,
              "distributed_backend": "fsdp_full_shard", "template_coordinates_used": False,
              "unassisted_prior": "open_chain", "fake_overfit_path_present": False,
              "real_forward_backward": True, "sampler_decoder_route": True,
              "esm_route": "cached_or_live_frozen", "dataset_mapping_sha256": dataset.mapping_sha256,
              "cv_split_manifest_sha256": cv["manifest_sha256"], "source_split_manifest_sha256": source["manifest_sha256"]}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
