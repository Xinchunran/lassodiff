"""Programmatic, template-free peptide priors for LassoDiff Mini."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

from .atom_schema_lasso import ATOM_CA, CandidateCondition
from .internal_coordinates import (
    CA_TRACE_SPACING, build_backbone_from_ca_trace, build_backbone_from_torsions,
    place_acceptor_reactive_atoms,
)


PriorMode = Literal["open_chain", "single_crossing", "topology_corrupted"]


@dataclass(frozen=True)
class PeptidePriorSample:
    coordinates: torch.Tensor
    atom_mask: torch.Tensor
    mode: str
    corruption: str | None = None


def _rand(shape, generator: torch.Generator | None, *, dtype=torch.float32):
    return torch.rand(shape, generator=generator, dtype=dtype)


def _random_rotation(generator: torch.Generator | None, dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.randn((3, 3), generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r)).clamp(min=-1, max=1)
    q = q * torch.where(signs == 0, torch.ones_like(signs), signs)
    if float(torch.linalg.det(q)) < 0:
        q[:, 0] *= -1
    return q


def _rigid_randomize(coordinates, mask, generator, noise_scale):
    valid = mask[..., None].to(coordinates.dtype)
    center = (coordinates * valid).sum((0, 1), keepdim=True) / valid.sum((0, 1), keepdim=True).clamp_min(1)
    output = (coordinates - center) @ _random_rotation(generator, coordinates.dtype)
    if noise_scale > 0:
        output = output + torch.randn(output.shape, generator=generator, dtype=output.dtype) * noise_scale * valid
    return output * valid


def _has_self_clash(core: torch.Tensor, mask: torch.Tensor, threshold: float = 1.25) -> bool:
    points = core[mask]
    residue = torch.arange(core.shape[0])[:, None].expand_as(mask)[mask]
    distance = torch.cdist(points, points)
    allowed = (residue[:, None] - residue[None, :]).abs() > 1
    allowed &= torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    return bool(((distance < threshold) & allowed).any())


def _sample_open(candidate: CandidateCondition, generator, max_attempts: int = 32):
    length = len(candidate.sequence)
    for _attempt in range(max_attempts):
        # Broad coil/Ramachandran mixture; deliberately not helix-heavy.
        phi = -180.0 + 360.0 * _rand((length,), generator)
        psi = -180.0 + 360.0 * _rand((length,), generator)
        omega = torch.full((length,), 180.0)
        for index in range(length - 1):
            if candidate.sequence[index + 1] == "P" and float(_rand((), generator)) < 0.08:
                omega[index] = 0.0
        core = build_backbone_from_torsions(candidate.sequence, phi, psi, omega)
        core = place_acceptor_reactive_atoms(core, candidate, formed=False)
        if not _has_self_clash(core, candidate.core_atom_mask):
            return core
    raise RuntimeError("failed to sample a self-avoiding open peptide prior")


def _ring_trace(candidate: CandidateCondition) -> torch.Tensor:
    ring_size = candidate.k + 1
    if ring_size < 4:
        raise ValueError("single-crossing prior requires at least four ring residues")
    radius = CA_TRACE_SPACING / (2.0 * math.sin(math.pi / ring_size))
    theta = torch.arange(ring_size, dtype=torch.float32) * (2.0 * math.pi / ring_size)
    return torch.stack((radius * torch.cos(theta), .82 * radius * torch.sin(theta), .18 * torch.sin(2 * theta)), dim=-1)


def _bridge(start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
    if steps < 1:
        return start.new_zeros((0, 3))
    direct = end - start
    distance = float(direct.norm())
    if distance > steps * CA_TRACE_SPACING:
        raise ValueError("candidate plug is unreachable from ring with peptide spacing")
    unit = direct / direct.norm().clamp_min(1e-8)
    normal = torch.linalg.cross(unit, torch.tensor([0.0, 0.0, 1.0]), dim=-1)
    if float(normal.norm()) < 1e-6:
        normal = torch.tensor([0.0, 1.0, 0.0])
    normal = normal / normal.norm().clamp_min(1e-8)
    if steps == 1:
        if not math.isclose(distance, CA_TRACE_SPACING, rel_tol=0, abs_tol=.05):
            raise ValueError("one-residue bridge must be one CA spacing")
        return end[None]
    fractions = torch.arange(steps + 1, dtype=start.dtype) / steps
    straight_step = distance / steps
    amplitude = math.sqrt(max(CA_TRACE_SPACING**2 - straight_step**2, 0.0)) * steps / math.pi
    points = start + fractions[:, None] * direct + amplitude * torch.sin(math.pi * fractions)[:, None] * normal
    # FABRIK makes every segment a peptide-like CA spacing while preserving
    # both endpoints.  It uses no target/template coordinates.
    for _ in range(80):
        points[-1] = end
        for index in range(steps - 1, -1, -1):
            vector = points[index] - points[index + 1]
            points[index] = points[index + 1] + CA_TRACE_SPACING * vector / vector.norm().clamp_min(1e-8)
        points[0] = start
        for index in range(steps):
            vector = points[index + 1] - points[index]
            points[index + 1] = points[index] + CA_TRACE_SPACING * vector / vector.norm().clamp_min(1e-8)
    if float((points[-1] - end).norm()) > .05:
        raise ValueError("failed to realize candidate bridge with peptide spacing")
    points[-1] = end
    return points[1:]


def _topological_ca_trace(candidate: CandidateCondition, corruption: str | None) -> torch.Tensor:
    length, k, p = len(candidate.sequence), candidate.k, candidate.p
    ca = torch.zeros((length, 3), dtype=torch.float32)
    ca[:k + 1] = _ring_trace(candidate)
    ring_center = ca[:k + 1].mean(0)
    edge = ca[k] - ring_center
    if corruption == "wrong_plug":
        crossing_start = min(p + 1, length - 2)
    else:
        crossing_start = p
    pre_steps = crossing_start - k
    # Edge-grazing is part of the endpoint geometry, not a post-hoc move of
    # the plug.  Moving the plug after `_bridge` was built could stretch the
    # final CA--CA segment well beyond peptide geometry for larger rings.
    inside_fraction = .96 if corruption == "edge_grazing" else .42
    radial_step = float(((1.0 - inside_fraction) * edge).norm())
    if pre_steps == 1 and radial_step >= CA_TRACE_SPACING:
        inside_fraction = 1.0 - 3.2 / float(edge.norm())
        radial_step = 3.2
    plug_z = 1.45 if pre_steps > 1 else math.sqrt(max(CA_TRACE_SPACING**2 - radial_step**2, .25))
    inside = ring_center + inside_fraction * edge
    plug = inside.clone()
    plug[2] = plug_z
    ca[k + 1:crossing_start + 1] = _bridge(ca[k], plug, pre_steps)

    if corruption == "no_crossing":
        for index in range(crossing_start + 1, length):
            direction = torch.tensor([1.0, .25 * (-1.0 if index % 2 else 1.0), 0.0])
            direction = direction / direction.norm()
            ca[index] = ca[index - 1] + CA_TRACE_SPACING * direction
            ca[index, 2] = max(float(ca[index, 2]), plug_z)
        return ca

    below = ca[crossing_start].clone()
    below[2] = -plug_z
    horizontal = math.sqrt(CA_TRACE_SPACING**2 - (2 * plug_z)**2)
    tangent = torch.tensor([-edge[1], edge[0], 0.0])
    tangent = tangent / tangent.norm().clamp_min(1e-8)
    below = below + horizontal * tangent
    ca[crossing_start + 1] = below

    next_index = crossing_start + 2
    if corruption == "double_crossing" and next_index < length:
        above = below.clone()
        above[2] = plug_z
        above = above + horizontal * tangent
        ca[next_index] = above
        next_index += 1
    for index in range(next_index, length):
        direction = torch.tensor([1.0, .3 * (-1.0 if index % 2 else 1.0), 0.0])
        direction = direction / direction.norm()
        ca[index] = ca[index - 1] + CA_TRACE_SPACING * direction
        if corruption != "double_crossing":
            ca[index, 2] = min(float(ca[index, 2]), -plug_z)
        else:
            ca[index, 2] = max(float(ca[index, 2]), plug_z)
    return ca


def _sample_topological(candidate, generator, corruption):
    ca = _topological_ca_trace(candidate, corruption)
    core = build_backbone_from_ca_trace(candidate.sequence, ca)
    return place_acceptor_reactive_atoms(core, candidate, formed=corruption != "no_crossing")


def sample_peptide_prior(
    candidate: CandidateCondition,
    *,
    mode: PriorMode,
    generator: torch.Generator | None = None,
    noise_scale: float = 0.05,
    corruption: str | None = None,
) -> PeptidePriorSample:
    """Sample a prior using only sequence, k/p and universal peptide geometry."""
    if mode == "open_chain":
        core = _sample_open(candidate, generator)
        corruption = None
    elif mode == "single_crossing":
        core = _sample_topological(candidate, generator, None)
        corruption = None
    elif mode == "topology_corrupted":
        variants = ("no_crossing", "double_crossing", "wrong_plug", "edge_grazing")
        if corruption is None:
            corruption = variants[int(torch.randint(len(variants), (), generator=generator))]
        if corruption not in variants:
            raise ValueError(f"unknown topology corruption: {corruption}")
        if corruption == "double_crossing" and candidate.p + 2 >= len(candidate.sequence):
            corruption = "no_crossing"
        core = _sample_topological(candidate, generator, corruption)
    else:
        raise ValueError(f"unknown peptide prior mode: {mode}")
    randomized = _rigid_randomize(core, candidate.core_atom_mask, generator, noise_scale)
    return PeptidePriorSample(randomized, candidate.core_atom_mask.clone(), mode, corruption)


def sample_prior_mode(
    generator: torch.Generator | None = None,
    *,
    open_chain: float = .40,
    single_crossing: float = .40,
    topology_corrupted: float = .20,
) -> PriorMode:
    weights = torch.tensor([open_chain, single_crossing, topology_corrupted], dtype=torch.float32)
    if bool((weights < 0).any()) or not math.isclose(float(weights.sum()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("prior probabilities must be non-negative and sum to one")
    index = int(torch.multinomial(weights, 1, generator=generator))
    return ("open_chain", "single_crossing", "topology_corrupted")[index]
