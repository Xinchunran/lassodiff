"""Versioned target/decoy construction for the V3 threading checker."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import torch

from .data.topology_targets import select_topology_target
from .topology_adapter import CandidateBatch
from .topology_threading import _hard_candidate, _reference_normal, _soft_candidate


THREADING_CHECKER_VERSION = "v3.2.p_tail_centroid_fan_v1"


def candidate_batch(candidate, *, device=None) -> CandidateBatch:
    device = device or torch.device("cpu")
    return CandidateBatch(
        torch.tensor([[int(candidate["k"])]], device=device),
        torch.tensor([[int(candidate["p"])]], device=device),
        torch.tensor([[int(candidate["acceptor_index"])]], device=device),
        torch.tensor([[float(candidate.get("prior", 1.0))]], device=device),
        torch.ones((1, 1), dtype=torch.bool, device=device),
    )


def unthreaded_decoy(ca: torch.Tensor, k: int, p: int) -> torch.Tensor:
    """Move the complete plug-to-tail curve away from its ring surface."""
    decoy = ca.clone()
    ring = decoy[:k + 1]
    thread = decoy[p:]
    if thread.numel() == 0:
        return decoy
    shift = ring[:, 0].max() - thread[:, 0].min() + 8.0
    decoy[p:, 0] += shift
    return decoy


def wrong_direction_decoy(ca: torch.Tensor, k: int, p: int) -> torch.Tensor:
    """Reflect the thread across the ordered ring's reference plane."""
    decoy = ca.clone()
    ring = decoy[:k + 1]
    normal = _reference_normal(ring)
    unit = normal / normal.norm().clamp_min(1e-8)
    center = ring.mean(dim=0)
    points = decoy[p:] - center
    decoy[p:] = center + points - 2.0 * (points * unit).sum(-1, keepdim=True) * unit
    return decoy


def _pairwise_auc(positive, negative) -> float:
    if not positive or not negative:
        return 0.0
    wins = sum(left > right for left in positive for right in negative)
    ties = sum(left == right for left in positive for right in negative)
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def build_threading_alignment_report(dataset, *, split_manifest_sha256: str = ""):
    """Build a deterministic JSON-serializable target/decoy alignment report."""
    from .data.lassopred_lmdb import _RECORD_PREFIX
    import pickle

    target_counts = Counter()
    target_scores, decoy_scores = [], []
    wrong_direction_flips = 0
    wrong_direction_checked = 0
    records = 0
    with dataset._open().begin() as txn:
        for record_id in dataset.record_ids:
            row = pickle.loads(txn.get(_RECORD_PREFIX + record_id.encode()))
            for candidate in row["candidates"]:
                try:
                    target, target_valid = select_topology_target(
                        row["conformers"], candidate, deterministic_index=0,
                    )
                except ValueError:
                    continue
                if not target_valid:
                    continue
                records += 1
                ca = target["coords"][:, 1].float()
                k, p = int(candidate["k"]), int(candidate["p"])
                hard = _hard_candidate(ca, k, p)
                target_counts["threaded" if hard[1] == 1 else "not_single_crossing"] += 1
                candidate_state = candidate_batch(candidate)
                score = float(_soft_candidate(ca, k, p, plane_temperature=.25,
                                               barycentric_temperature=.05,
                                               segment_temperature=.05))
                decoy = unthreaded_decoy(ca, k, p)
                decoy_scores.append(abs(float(_soft_candidate(
                    decoy, k, p, plane_temperature=.25,
                    barycentric_temperature=.05, segment_temperature=.05,
                ))))
                if hard[1] == 1:
                    target_scores.append(abs(score))
                reflected = wrong_direction_decoy(ca, k, p)
                reflected_hard = _hard_candidate(reflected, k, p)
                if hard[1] == 1:
                    wrong_direction_checked += 1
                    wrong_direction_flips += int(
                        reflected_hard[1] == 1 and reflected_hard[2] == -hard[2]
                    )
                _ = candidate_state  # keeps candidate schema construction tested here
    report = {
        "checker_version": THREADING_CHECKER_VERSION,
        "split_manifest_sha256": split_manifest_sha256,
        "chemistry_qualified_target_count": records,
        "target_counts": dict(sorted(target_counts.items())),
        "threaded_target_rate": target_counts["threaded"] / max(records, 1),
        "unthreaded_decoy_count": len(decoy_scores),
        "wrong_direction_checked": wrong_direction_checked,
        "wrong_direction_class_flip_rate": wrong_direction_flips / max(wrong_direction_checked, 1),
        "soft_target_count": len(target_scores),
        "soft_pairwise_auc_unthreaded": _pairwise_auc(target_scores, decoy_scores),
        "soft_target_median": statistics.median(target_scores) if target_scores else 0.0,
        "soft_decoy_median": statistics.median(decoy_scores) if decoy_scores else 0.0,
        "geometry_definition": "ordered CA ring 0..k, centroid-fan surface, plug-inclusive p..tail",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def write_threading_alignment_report(report, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_threading_alignment_report(report, *, min_target_rate: float = .90, min_auc: float = .90):
    if report.get("checker_version") != THREADING_CHECKER_VERSION:
        raise RuntimeError("threading alignment checker version mismatch")
    if report.get("chemistry_qualified_target_count", 0) < 1:
        raise RuntimeError("threading truth set is empty")
    if report.get("threaded_target_rate", 0.0) < min_target_rate:
        raise RuntimeError("threading target rate is below the signed release threshold")
    if report.get("soft_pairwise_auc_unthreaded", 0.0) < min_auc:
        raise RuntimeError("threading surrogate decoy AUROC is below the release threshold")
    if report.get("wrong_direction_checked", 0) and report.get("wrong_direction_class_flip_rate", 0.0) < .9:
        raise RuntimeError("wrong-direction decoy class flip rate is below the release threshold")
    unhashed = {key: value for key, value in report.items() if key != "report_sha256"}
    expected = hashlib.sha256(json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if report.get("report_sha256") != expected:
        raise RuntimeError("threading alignment report hash mismatch")
    return report
