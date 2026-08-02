"""Chemistry-qualified candidate targets for V3 structure supervision.

The processed seven-atom schema stores ``N/CA/C/O/Ciso/O/unused`` for a
formed isopeptide amide.  The second carboxylate oxygen is absent because the
N-terminal nitrogen replaces it.  Target qualification must therefore use
the available carbonyl oxygen and must never require atom slot six.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import torch

from ..losses_lasso import ATOM_CA, ATOM_CISO, ATOM_N, ATOM_O1, ATOM_O2


@dataclass(frozen=True)
class TopologyTargetQuality:
    valid: bool
    iso_distance: float
    iso_angle_degrees: float
    iso_plane_distance: float
    has_second_oxygen: bool


def _angle(left: torch.Tensor, center: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    first, second = left - center, right - center
    cosine = (first * second).sum() / (first.norm() * second.norm()).clamp_min(1e-8)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def assess_topology_target(
    conformer: Mapping[str, Any], candidate: Mapping[str, Any],
) -> TopologyTargetQuality:
    coords = conformer["coords"].float()
    atom_mask = conformer["atom_mask"].bool()
    if coords.ndim != 3 or atom_mask.shape != coords.shape[:-1] or coords.shape[1] < 7:
        raise ValueError("topology target must use the seven-atom [L,A,3] schema")
    acceptor = int(candidate["acceptor_index"])
    if not 0 <= acceptor < coords.shape[0]:
        raise ValueError("candidate acceptor is outside topology target")
    available = bool(
        atom_mask[0, ATOM_N]
        and atom_mask[acceptor, ATOM_CA]
        and atom_mask[acceptor, ATOM_CISO]
        and atom_mask[acceptor, ATOM_O1]
    )
    nterm = coords[0, ATOM_N]
    ciso = coords[acceptor, ATOM_CISO]
    oxygen = coords[acceptor, ATOM_O1]
    acceptor_ca = coords[acceptor, ATOM_CA]
    distance = (nterm - ciso).norm()
    angle = _angle(nterm, ciso, oxygen)
    # The fixed schema omits the ASP-CB/GLU-CG parent atom.  CA is an explicit,
    # stable proxy used both for target qualification and generated metrics.
    normal = torch.linalg.cross(oxygen - ciso, acceptor_ca - ciso, dim=-1)
    plane = ((nterm - ciso) * normal).sum().abs() / normal.norm().clamp_min(1e-8)
    finite = bool(torch.isfinite(torch.stack((distance, angle, plane))).all())
    valid = bool(
        available and finite
        and 1.1 <= float(distance) <= 1.7
        and 90.0 <= float(angle) <= 150.0
        and float(plane) <= 0.5
    )
    return TopologyTargetQuality(
        valid=valid,
        iso_distance=float(distance),
        iso_angle_degrees=float(angle),
        iso_plane_distance=float(plane),
        has_second_oxygen=bool(atom_mask[acceptor, ATOM_O2]),
    )


def _matching_rank(conformers: Sequence[Mapping[str, Any]], rank: int):
    pattern = re.compile(rf"(?:min|relax){int(rank)}")
    return [item for item in conformers if pattern.fullmatch(str(item["name"]))]


def select_topology_target(
    conformers: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any], *,
    deterministic_index: int,
):
    """Prefer a valid relaxed target, then a valid minimum.

    If no valid topology label exists, return a same-rank fallback together
    with ``False``.  The V3 collate path preserves that explicit target-valid
    mask and never clamps or relabels the candidate.
    """
    available = _matching_rank(conformers, int(candidate["rank"]))
    if not available:
        raise ValueError(f"candidate rank {candidate['rank']} has no matching target")
    qualified = [item for item in available if assess_topology_target(item, candidate).valid]
    relaxed = [item for item in qualified if str(item["name"]).startswith("relax")]
    pool = relaxed or qualified
    if pool:
        return pool[int(deterministic_index) % len(pool)], True
    return available[int(deterministic_index) % len(available)], False


def record_has_topology_target(record: Mapping[str, Any]) -> bool:
    for candidate in record["candidates"]:
        _selected, valid = select_topology_target(
            record["conformers"], candidate, deterministic_index=0,
        )
        if valid:
            return True
    return False
