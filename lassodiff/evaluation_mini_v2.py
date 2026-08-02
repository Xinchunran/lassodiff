"""Fail-closed paired evaluation schema for mini_dev."""
from __future__ import annotations

REQUIRED_BASELINE_KEYS = ("prior_only_unassisted", "untrained_model_unassisted", "trained_model_unassisted", "trained_model_assisted")
REQUIRED_SYSTEMS = REQUIRED_BASELINE_KEYS
REQUIRED_SAMPLE_FIELDS = frozenset({"finite", "strict_valid", "backbone_valid", "formed_geometry", "crossing_count", "plug_match", "tail_persistence", "clash_valid", "ca_rmsd_best_target", "lddt_best_target"})


def evaluate_rollout_samples(core_samples, target_core, target_mask, candidate):
    """Evaluate every generated sample through the strict checker."""
    import torch
    from .metrics_mini_v2 import best_ca_rmsd, lddt_score
    rows = []
    valid_targets = target_core[target_mask.any(dim=(-1, -2))] if target_mask.ndim == 3 else target_core
    best = best_ca_rmsd(core_samples, valid_targets)
    for sample in core_samples:
        ca = sample[:, 1, :]
        target_ca = valid_targets[:, :, 1, :]
        lddt = max(float(lddt_score(ca, t)) for t in target_ca) if len(target_ca) else 0.0
        mask = torch.ones(sample.shape[:-1], dtype=torch.bool, device=sample.device)
        rows.append(evaluate_generated_candidate(sample, mask, candidate,
                                                  ca_rmsd_best_target=float(best), lddt_best_target=lddt))
    return rows


def evaluate_generated_candidate(core_coordinates, core_atom_mask, candidate, *, atom14_coordinates=None,
                                 atom14_atom_mask=None, ca_rmsd_best_target=None, lddt_best_target=None) -> dict:
    """Build the required sample row through the unchanged authoritative checker."""
    from .validation.strict_lasso import strict_lasso_check
    import torch

    finite = bool(torch.isfinite(core_coordinates[core_atom_mask]).all())
    if atom14_coordinates is not None and atom14_atom_mask is not None:
        finite = finite and bool(torch.isfinite(atom14_coordinates[atom14_atom_mask]).all())
    result = strict_lasso_check(core_coordinates, core_atom_mask, candidate,
                                atom14_coordinates=atom14_coordinates,
                                atom14_atom_mask=atom14_atom_mask)
    return {
        "finite": finite,
        "strict_valid": bool(result.valid and finite),
        "backbone_valid": bool(result.backbone_valid),
        "formed_geometry": bool(result.formed_geometry_valid),
        "crossing_count": int(result.crossing_count),
        "plug_match": bool(result.plug_match),
        "tail_persistence": bool(result.tail_persistence),
        "clash_valid": bool(result.clash_valid),
        "ca_rmsd_best_target": float(ca_rmsd_best_target) if ca_rmsd_best_target is not None else float("nan"),
        "lddt_best_target": float(lddt_best_target) if lddt_best_target is not None else float("nan"),
        "rejection_reasons": list(result.rejection_reasons),
    }


def validate_evaluation_report(report: dict) -> None:
    if "systems" in report:
        systems = report.get("systems", {})
        missing = set(REQUIRED_SYSTEMS) - set(systems)
        if missing:
            raise ValueError(f"evaluation report is missing systems: {sorted(missing)}")
        sample_count = report.get("sample_count")
        if not isinstance(report.get("seed_set"), list) or sample_count is None:
            raise ValueError("paired evaluation requires seed_set and sample_count")
        for name in REQUIRED_SYSTEMS:
            samples = systems[name].get("samples")
            if not isinstance(samples, list) or len(samples) != sample_count:
                raise ValueError(f"system {name} has an invalid sample list")
            for sample in samples:
                missing_fields = REQUIRED_SAMPLE_FIELDS - set(sample)
                if missing_fields:
                    raise ValueError(f"sample in {name} is missing {sorted(missing_fields)}")
        return
    missing = set(REQUIRED_BASELINE_KEYS) - set(report)
    if missing:
        raise ValueError(f"evaluation report is missing baselines: {sorted(missing)}")
    for name in REQUIRED_BASELINE_KEYS:
        row = report[name]
        if "strict_valid_rate" not in row or "sample_count" not in row:
            raise ValueError(f"baseline {name} lacks strict-valid summary")
