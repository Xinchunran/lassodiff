"""Self-contained full-tail hard checker and soft exactly-one surrogate."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MiniThreadingCheck:
    valid: torch.Tensor
    crossing_count: torch.Tensor
    observed_crossing_segment: torch.Tensor
    plug_consistent: torch.Tensor
    threading_valid: torch.Tensor


@dataclass(frozen=True)
class MiniTopologySurrogate:
    expected_count: torch.Tensor
    probability_exactly_one: torch.Tensor
    valid: torch.Tensor


def _triangles(ring):
    center = ring.mean(0)
    return torch.stack((center.expand_as(ring), ring, torch.roll(ring, -1, 0)), dim=1)


def _normal(ring):
    center = ring.mean(0)
    return torch.linalg.cross(ring - center, torch.roll(ring, -1, 0) - center, dim=-1).sum(0)


def _hard_candidate(ca, length, k, p, eps=1e-7):
    ring, tail = ca[:k + 1].float(), ca[k + 1:length].float()
    if ring.shape[0] < 3 or tail.shape[0] < 2 or not bool(torch.isfinite(ring).all() and torch.isfinite(tail).all()):
        return None
    triangles, reference = _triangles(ring), _normal(ring)
    if float(reference.norm()) <= eps:
        return None
    hits = []
    for segment in range(tail.shape[0] - 1):
        start, end = tail[segment], tail[segment + 1]
        direction = end - start
        if float(direction.norm()) <= eps:
            continue
        for triangle in triangles:
            a, b, c = triangle
            edge1, edge2 = b - a, c - a
            h = torch.linalg.cross(direction, edge2, dim=-1)
            determinant = torch.dot(edge1, h)
            if abs(float(determinant)) <= eps:
                continue
            inverse = 1.0 / determinant
            offset = start - a
            u = torch.dot(offset, h) * inverse
            q = torch.linalg.cross(offset, edge1, dim=-1)
            v = torch.dot(direction, q) * inverse
            fraction = torch.dot(edge2, q) * inverse
            if (
                -eps <= float(u) <= 1 + eps and -eps <= float(v)
                and float(u + v) <= 1 + eps and .05 < float(fraction) < .95
            ):
                point = start + fraction * direction
                edge_distance = torch.stack([
                    _point_segment_distance(point, ring[index], ring[(index + 1) % ring.shape[0]])
                    for index in range(ring.shape[0])
                ]).min()
                if float(edge_distance) >= .05:
                    hits.append((segment, float(fraction)))
    unique = []
    for hit in sorted(hits):
        if unique and hit[0] == unique[-1][0] and abs(hit[1] - unique[-1][1]) <= 1e-5:
            continue
        unique.append(hit)
    count = len(unique)
    observed = k + 1 + unique[0][0] if count == 1 else -1
    persistence = False
    if count == 1:
        segment = unique[0][0]
        reference = reference / reference.norm().clamp_min(eps)
        center = ring.mean(0)
        destination = torch.sign(torch.dot(tail[segment + 1] - center, reference))
        remaining = tail[segment + 1:]
        persistence = bool((((remaining - center) @ reference) * destination > 0).float().mean() >= .8)
    return count, observed, observed == p, persistence


def _point_segment_distance(point, start, end):
    edge = end - start
    fraction = torch.dot(point - start, edge) / edge.square().sum().clamp_min(1e-8)
    return (point - (start + fraction.clamp(0, 1) * edge)).norm()


@torch.no_grad()
def hard_threading_check_ca(ca, token_mask, candidates, ca_mask=None):
    if ca.ndim != 4 or ca.shape[:2] != candidates.k.shape or token_mask.shape != (ca.shape[0], ca.shape[2]):
        raise ValueError("Mini threading checker input shape mismatch")
    B, M, L, _ = ca.shape
    candidates.validate(L)
    valid = torch.zeros((B, M), dtype=torch.bool, device=ca.device)
    count = torch.zeros((B, M), dtype=torch.long, device=ca.device)
    observed = torch.full((B, M), -1, dtype=torch.long, device=ca.device)
    plug = torch.zeros((B, M), dtype=torch.bool, device=ca.device)
    persistent = torch.zeros((B, M), dtype=torch.bool, device=ca.device)
    for b in range(B):
        length = int(token_mask[b].sum())
        for m in range(M):
            if not bool(candidates.candidate_mask[b, m]):
                continue
            k, p = int(candidates.k[b, m]), int(candidates.p[b, m])
            if ca_mask is not None and not bool(ca_mask[b, m, :length].all()):
                continue
            result = _hard_candidate(ca[b, m], length, k, p)
            if result is None:
                continue
            valid[b, m] = True
            count[b, m], observed[b, m], plug[b, m], persistent[b, m] = result
    return MiniThreadingCheck(valid, count, observed, plug, valid & count.eq(1) & persistent)


def _soft_candidate(ca, length, k, eps=1e-6):
    ring, tail = ca[:k + 1].float(), ca[k + 1:length].float()
    if ring.shape[0] < 3 or tail.shape[0] < 2 or float(_normal(ring).norm()) <= eps:
        return None
    triangles = _triangles(ring)
    start, end = tail[:-1], tail[1:]
    direction = end - start
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1, edge2 = b - a, c - a
    normal = torch.linalg.cross(edge1, edge2, dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(eps)
    d0 = ((start[:, None] - a[None]) * normal[None]).sum(-1)
    d1 = ((end[:, None] - a[None]) * normal[None]).sum(-1)
    denominator = d0 - d1
    safe = torch.where(denominator >= 0, denominator.clamp_min(eps), denominator.clamp_max(-eps))
    fraction = d0 / safe
    point = start[:, None] + fraction[..., None] * direction[:, None]
    relative = point - a[None]
    dot00, dot01, dot11 = edge1.square().sum(-1), (edge1 * edge2).sum(-1), edge2.square().sum(-1)
    dot20, dot21 = (relative * edge1[None]).sum(-1), (relative * edge2[None]).sum(-1)
    divisor = (dot00 * dot11 - dot01.square()).clamp_min(eps)
    u = (dot11[None] * dot20 - dot01[None] * dot21) / divisor[None]
    v = (dot00[None] * dot21 - dot01[None] * dot20) / divisor[None]
    w = 1 - u - v
    p_triangle = (
        torch.sigmoid((-d0 * d1 - .0025) / .03)
        * torch.sigmoid((fraction - .05) / .03) * torch.sigmoid((.95 - fraction) / .03)
        * torch.sigmoid(u / .03) * torch.sigmoid(v / .03) * torch.sigmoid(w / .03)
    ).clamp(eps, 1 - eps)
    return 1 - torch.exp(torch.log1p(-p_triangle).sum(-1))


def soft_topology_surrogate_ca(ca, token_mask, candidates, ca_mask=None):
    if ca.ndim != 4 or ca.shape[:2] != candidates.k.shape:
        raise ValueError("Mini topology surrogate input shape mismatch")
    B, M, L, _ = ca.shape
    candidates.validate(L)
    expected_rows, one_rows, valid_rows = [], [], []
    for b in range(B):
        expected_row, one_row, valid_row = [], [], []
        length = int(token_mask[b].sum())
        for m in range(M):
            available = ca_mask is None or bool(ca_mask[b, m, :length].all())
            result = None if not bool(candidates.candidate_mask[b, m]) or not available else _soft_candidate(
                ca[b, m], length, int(candidates.k[b, m]),
            )
            if result is None:
                zero = ca[b, m].sum() * 0
                expected_row.append(zero); one_row.append(zero); valid_row.append(False)
            else:
                p_zero, p_one = torch.ones_like(result[0]), torch.zeros_like(result[0])
                for probability in result:
                    p_one, p_zero = p_one * (1 - probability) + p_zero * probability, p_zero * (1 - probability)
                expected_row.append(result.sum()); one_row.append(p_one); valid_row.append(True)
        expected_rows.append(torch.stack(expected_row)); one_rows.append(torch.stack(one_row)); valid_rows.append(valid_row)
    return MiniTopologySurrogate(
        torch.stack(expected_rows), torch.stack(one_rows), torch.tensor(valid_rows, dtype=torch.bool, device=ca.device),
    )
