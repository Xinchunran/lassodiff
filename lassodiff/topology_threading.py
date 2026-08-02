"""Shared hard checker and differentiable threading geometry.

Both paths use the same geometric objects: the ordered CA ring ``0..k``
closed by an explicit centroid-fan surface, and the candidate thread path
from plug ``p`` through the last valid residue.  Keeping this definition in
one module prevents the historical bug where evaluation checked the pre-plug
loop while training skipped the decisive ``p -> p+1`` segment.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThreadingCheck:
    valid: torch.Tensor
    crossing_count: torch.Tensor
    signed_crossing: torch.Tensor
    threading_class: torch.Tensor
    confidence: torch.Tensor


def _surface_triangles(ring: torch.Tensor) -> torch.Tensor:
    center = ring.mean(dim=0)
    following = torch.roll(ring, shifts=-1, dims=0)
    return torch.stack((center.expand_as(ring), ring, following), dim=1)


def _reference_normal(ring: torch.Tensor) -> torch.Tensor:
    center = ring.mean(dim=0)
    following = torch.roll(ring, shifts=-1, dims=0)
    return torch.linalg.cross(ring - center, following - center, dim=-1).sum(dim=0)


def _hard_candidate(ca: torch.Tensor, k: int, p: int, *, eps: float = 1e-7):
    ring = ca[:k + 1].float()
    thread = ca[p:].float()
    if ring.shape[0] < 3 or thread.shape[0] < 2:
        return False, 0, 0, 0.0
    if not bool(torch.isfinite(ring).all() and torch.isfinite(thread).all()):
        return False, 0, 0, 0.0
    triangles = _surface_triangles(ring)
    reference_normal = _reference_normal(ring)
    hits: list[tuple[int, float, int, float]] = []
    surface_area = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1,
    ).norm(dim=-1).sum()
    if float(surface_area) <= eps or float(reference_normal.norm()) <= eps:
        return False, 0, 0, 0.0
    for segment_index in range(thread.shape[0] - 1):
        start, end = thread[segment_index], thread[segment_index + 1]
        direction = end - start
        if float(direction.norm()) <= eps:
            continue
        for triangle in triangles:
            a, b, c = triangle
            edge1, edge2 = b - a, c - a
            normal = torch.linalg.cross(edge1, edge2, dim=-1)
            h = torch.linalg.cross(direction, edge2, dim=-1)
            determinant = torch.dot(edge1, h)
            if abs(float(determinant)) <= eps:
                continue
            inverse = 1.0 / determinant
            offset = start - a
            u = torch.dot(offset, h) * inverse
            q = torch.linalg.cross(offset, edge1, dim=-1)
            v = torch.dot(direction, q) * inverse
            t = torch.dot(edge2, q) * inverse
            if not (
                -eps <= float(u) <= 1.0 + eps
                and -eps <= float(v)
                and float(u + v) <= 1.0 + eps
                and eps < float(t) < 1.0 - eps
            ):
                continue
            sign = 1 if float(torch.dot(direction, reference_normal)) > 0 else -1
            margin = max(0.0, min(float(u), float(v), float(1.0 - u - v)))
            confidence = min(1.0, margin / 0.1) * min(1.0, min(float(t), float(1.0 - t)) / 0.1)
            hits.append((segment_index, float(t), sign, confidence))
    # A hit on a fan edge is reported by both adjacent triangles.  Deduplicate
    # by segment and interpolation coordinate before counting topology.
    unique: list[tuple[int, float, int, float]] = []
    for hit in sorted(hits, key=lambda item: (item[0], item[1])):
        if unique and hit[0] == unique[-1][0] and abs(hit[1] - unique[-1][1]) <= 1e-5:
            if hit[3] > unique[-1][3]:
                unique[-1] = hit
            continue
        unique.append(hit)
    count = len(unique)
    signed = sum(item[2] for item in unique)
    confidence = min((item[3] for item in unique), default=0.0)
    return True, count, signed, confidence


@torch.no_grad()
def hard_threading_check_ca(
    ca: torch.Tensor, token_mask: torch.Tensor, candidates, ca_mask: torch.Tensor | None = None,
) -> ThreadingCheck:
    """Authoritative segment/surface intersections for candidate ``p..tail``."""
    if ca.ndim != 4 or ca.shape[:2] != candidates.k.shape or token_mask.shape != (ca.shape[0], ca.shape[2]):
        raise ValueError("threading checker requires CA [B,M,L,3] and token mask [B,L]")
    if ca_mask is not None and ca_mask.shape != ca.shape[:-1]:
        raise ValueError("threading checker CA mask must have shape [B,M,L]")
    B, M, L, _ = ca.shape
    candidates.validate(L)
    valid = torch.zeros((B, M), dtype=torch.bool, device=ca.device)
    count = torch.zeros((B, M), dtype=torch.long, device=ca.device)
    signed = torch.zeros((B, M), dtype=torch.long, device=ca.device)
    confidence = torch.zeros((B, M), dtype=torch.float32, device=ca.device)
    for b in range(B):
        length = int(token_mask[b].sum())
        for m in range(M):
            if not bool(candidates.candidate_mask[b, m]):
                continue
            k, p = int(candidates.k[b, m]), int(candidates.p[b, m])
            atoms_available = ca_mask is None or bool(ca_mask[b, m, :length].all())
            if p + 1 >= length or not bool(token_mask[b, :length].all()) or not atoms_available:
                continue
            ok, crossings, direction, score = _hard_candidate(ca[b, m], k, p)
            valid[b, m] = ok
            count[b, m] = crossings
            signed[b, m] = direction
            confidence[b, m] = score
    topology_class = signed.sign()
    return ThreadingCheck(valid, count, signed, topology_class, confidence)


def _soft_candidate(
    ca: torch.Tensor, k: int, p: int, *, plane_temperature: float,
    barycentric_temperature: float, segment_temperature: float,
) -> torch.Tensor:
    ring = ca[:k + 1].float()
    thread = ca[p:].float()
    if ring.shape[0] < 3 or thread.shape[0] < 2:
        return ca.sum() * 0.0
    triangles = _surface_triangles(ring)
    reference_normal = _reference_normal(ring)
    reference_unit = reference_normal / reference_normal.norm().clamp_min(1e-6)
    start, end = thread[:-1], thread[1:]
    direction = end - start                                      # [S,3]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]  # [R,3]
    edge1, edge2 = b - a, c - a
    normal = torch.linalg.cross(edge1, edge2, dim=-1)
    unit_normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    d0 = ((start[:, None] - a[None]) * unit_normal[None]).sum(-1)  # [S,R]
    d1 = ((end[:, None] - a[None]) * unit_normal[None]).sum(-1)
    denominator = d0 - d1
    safe_denominator = denominator.sign().masked_fill(denominator == 0, 1.0) * denominator.abs().clamp_min(1e-6)
    fraction = d0 / safe_denominator
    point = start[:, None] + fraction[..., None] * direction[:, None]
    relative = point - a[None]
    dot00 = edge1.square().sum(-1)
    dot01 = (edge1 * edge2).sum(-1)
    dot11 = edge2.square().sum(-1)
    dot20 = (relative * edge1[None]).sum(-1)
    dot21 = (relative * edge2[None]).sum(-1)
    bary_denominator = (dot00 * dot11 - dot01.square()).clamp_min(1e-8)
    u = (dot11[None] * dot20 - dot01[None] * dot21) / bary_denominator[None]
    v = (dot00[None] * dot21 - dot01[None] * dot20) / bary_denominator[None]
    inside = (
        torch.sigmoid(u / barycentric_temperature)
        * torch.sigmoid(v / barycentric_temperature)
        * torch.sigmoid((1.0 - u - v) / barycentric_temperature)
    )
    on_segment = torch.sigmoid(fraction / segment_temperature) * torch.sigmoid(
        (1.0 - fraction) / segment_temperature
    )
    opposite_sides = torch.sigmoid(-(d0 * d1) / (plane_temperature * plane_temperature))
    hit_probabilities = (inside * on_segment * opposite_sides).clamp(0.0, 1.0 - 1e-6)
    # Adjacent fan triangles may both own a boundary hit.  A soft union across
    # triangles counts that geometric event once per chain segment.
    hit_probability = 1.0 - (1.0 - hit_probabilities).prod(dim=-1)
    orientation = torch.tanh(
        (direction * reference_unit).sum(-1)
        / (plane_temperature * direction.norm(dim=-1).clamp_min(1e-6))
    )
    return (hit_probability * orientation).sum()


def soft_threading_score_ca(
    ca: torch.Tensor, token_mask: torch.Tensor, candidates, ca_mask: torch.Tensor | None = None, *,
    plane_temperature: float = 0.25, barycentric_temperature: float = 0.05,
    segment_temperature: float = 0.05,
) -> torch.Tensor:
    """Differentiable signed crossing score using the checker's geometry."""
    if ca.ndim != 4 or ca.shape[:2] != candidates.k.shape or token_mask.shape != (ca.shape[0], ca.shape[2]):
        raise ValueError("threading surrogate requires CA [B,M,L,3] and token mask [B,L]")
    if ca_mask is not None and ca_mask.shape != ca.shape[:-1]:
        raise ValueError("threading surrogate CA mask must have shape [B,M,L]")
    B, M, L, _ = ca.shape
    candidates.validate(L)
    rows = []
    for b in range(B):
        values = []
        length = int(token_mask[b].sum())
        for m in range(M):
            k, p = int(candidates.k[b, m]), int(candidates.p[b, m])
            atoms_available = ca_mask is None or bool(ca_mask[b, m, :length].all())
            if not bool(candidates.candidate_mask[b, m]) or p + 1 >= length or not atoms_available:
                values.append(ca[b, m].sum() * 0.0)
            else:
                values.append(_soft_candidate(
                    ca[b, m], k, p, plane_temperature=plane_temperature,
                    barycentric_temperature=barycentric_temperature,
                    segment_temperature=segment_temperature,
                ))
        rows.append(torch.stack(values))
    return torch.stack(rows)
