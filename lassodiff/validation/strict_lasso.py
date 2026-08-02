"""Strict, projection-free chemistry/topology checker for Mini outputs."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..atom_schema_lasso import (
    ATOM_C, ATOM_CA, ATOM_CB, ATOM_CISO, ATOM_N, ATOM_OISO, CandidateCondition,
)
from ..sidechain_builder import atom14_names
from ..topology_adapter import CandidateBatch
from .threading_mini import hard_threading_check_ca


@dataclass(frozen=True)
class StrictLassoResult:
    valid: bool
    rejection_reasons: tuple[str, ...]
    iso_distance: float
    oxygen_angle: float
    side_angle: float
    amide_plane_distance: float
    crossing_count: int
    observed_crossing_segment: int
    plug_match: bool
    backbone_valid: bool
    clash_valid: bool


def _angle(a, center, b) -> float:
    left, right = a - center, b - center
    cosine = torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-8)
    return float(torch.rad2deg(torch.acos(cosine.clamp(-1, 1))))


def _side_anchor(core, candidate, atom14_coordinates, atom14_atom_mask):
    if candidate.sequence[candidate.k] == "D":
        return core[candidate.k, ATOM_CB]
    if atom14_coordinates is None or atom14_atom_mask is None:
        return None
    names = atom14_names(candidate.sequence, candidate)[candidate.k]
    slot = names.index("CG")
    return atom14_coordinates[candidate.k, slot] if bool(atom14_atom_mask[candidate.k, slot]) else None


@torch.no_grad()
def strict_lasso_check(
    core_coordinates: torch.Tensor,
    core_atom_mask: torch.Tensor,
    candidate: CandidateCondition,
    *,
    atom14_coordinates: torch.Tensor | None = None,
    atom14_atom_mask: torch.Tensor | None = None,
) -> StrictLassoResult:
    if core_coordinates.shape != (len(candidate.sequence), 7, 3) or core_atom_mask.shape != core_coordinates.shape[:-1]:
        raise ValueError("strict checker requires core [L,7,3] and mask [L,7]")
    reasons: list[str] = []
    if not bool(torch.isfinite(core_coordinates[core_atom_mask]).all()):
        reasons.append("non_finite_core")
    required = candidate.core_atom_mask.to(core_atom_mask.device)
    if not bool(core_atom_mask[required].all()):
        reasons.append("missing_core_atom")

    nterm = core_coordinates[0, ATOM_N]
    ciso = core_coordinates[candidate.k, ATOM_CISO]
    oiso = core_coordinates[candidate.k, ATOM_OISO]
    side = _side_anchor(core_coordinates, candidate, atom14_coordinates, atom14_atom_mask)
    iso_distance = float((nterm - ciso).norm())
    oxygen_angle = _angle(oiso, ciso, nterm)
    side_angle = float("nan") if side is None else _angle(side, ciso, nterm)
    plane = float("nan")
    if side is None:
        reasons.append("missing_glu_cg")
    else:
        normal = torch.linalg.cross(oiso - ciso, side - ciso, dim=-1)
        plane = float(torch.dot(nterm - ciso, normal).abs() / normal.norm().clamp_min(1e-8))
    if not 1.20 <= iso_distance <= 1.70:
        reasons.append("iso_distance")
    if not 100.0 <= oxygen_angle <= 140.0:
        reasons.append("iso_oxygen_angle")
    if side is not None and not 100.0 <= side_angle <= 140.0:
        reasons.append("iso_side_angle")
    if side is not None and not plane <= .50:
        reasons.append("iso_amide_plane")

    peptide = (core_coordinates[:-1, ATOM_C] - core_coordinates[1:, ATOM_N]).norm(dim=-1)
    n_ca = (core_coordinates[:, ATOM_N] - core_coordinates[:, ATOM_CA]).norm(dim=-1)
    ca_c = (core_coordinates[:, ATOM_CA] - core_coordinates[:, ATOM_C]).norm(dim=-1)
    backbone_valid = bool(
        ((peptide >= 1.15) & (peptide <= 1.50)).all()
        and ((n_ca >= 1.30) & (n_ca <= 1.62)).all()
        and ((ca_c >= 1.35) & (ca_c <= 1.70)).all()
    )
    if not backbone_valid:
        reasons.append("backbone_bonds")

    length = len(candidate.sequence)
    batch_candidate = CandidateBatch(
        torch.tensor([[candidate.k]]), torch.tensor([[candidate.p]]), torch.tensor([[candidate.k]]),
        torch.ones((1, 1)), torch.ones((1, 1), dtype=torch.bool),
    )
    topology = hard_threading_check_ca(
        core_coordinates[:, ATOM_CA][None, None].cpu(), torch.ones((1, length), dtype=torch.bool),
        batch_candidate, core_atom_mask[:, ATOM_CA][None, None].cpu(),
    )
    count = int(topology.crossing_count[0, 0])
    observed = int(topology.observed_crossing_segment[0, 0])
    plug_match = bool(topology.plug_consistent[0, 0])
    if not bool(topology.valid[0, 0]):
        reasons.append("topology_not_evaluable")
    elif count != 1:
        reasons.append("crossing_not_exactly_one")
    if count == 1 and not plug_match:
        reasons.append("plug_mismatch")
    if count == 1 and not bool(topology.threading_valid[0, 0]):
        reasons.append("tail_not_persistent")

    coordinates = atom14_coordinates if atom14_coordinates is not None else core_coordinates
    mask = atom14_atom_mask if atom14_atom_mask is not None else core_atom_mask
    points = coordinates[mask]
    residue_ids = torch.arange(length)[:, None].expand_as(mask)[mask]
    distances = torch.cdist(points.float(), points.float())
    allowed = (residue_ids[:, None] - residue_ids[None, :]).abs() > 1
    allowed &= torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    clash_valid = not bool(((distances < 1.20) & allowed).any())
    if not clash_valid:
        reasons.append("severe_clash")
    return StrictLassoResult(
        not reasons, tuple(reasons), iso_distance, oxygen_angle, side_angle, plane,
        count, observed, plug_match, backbone_valid, clash_valid,
    )
