"""Differentiable peptide-chain reconstruction from backbone torsions."""
from __future__ import annotations

import math

import torch

from .atom_schema_lasso import ATOM_C, ATOM_CA, ATOM_CB, ATOM_N, ATOM_O


BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329
BOND_C_O = 1.231
BOND_CA_CB = 1.522


def _angle(value: torch.Tensor | float, degrees: float = False) -> torch.Tensor:
    value = torch.as_tensor(value)
    return value * (math.pi / 180.0) if degrees else value


def place_atom(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, length: float,
               bond_angle: torch.Tensor | float, dihedral: torch.Tensor | float,
               *, degrees: bool = False) -> torch.Tensor:
    """Place D after A-B-C while retaining gradients through angles."""
    angle = _angle(bond_angle, degrees)
    torsion = _angle(dihedral, degrees)
    bc = c - b
    bc = bc / bc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    normal = torch.linalg.cross(b - a, bc, dim=-1)
    fallback = torch.zeros_like(bc)
    fallback[..., 2] = 1.0
    alternate = torch.zeros_like(bc)
    alternate[..., 1] = 1.0
    near_parallel = normal.norm(dim=-1, keepdim=True) < 1e-7
    reference = torch.where((bc[..., 2:3].abs() > 0.9), alternate, fallback)
    normal = torch.where(near_parallel, torch.linalg.cross(reference, bc, dim=-1), normal)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    in_plane = torch.linalg.cross(normal, bc, dim=-1)
    direction = (-torch.cos(angle)[..., None] * bc
                 + torch.sin(angle)[..., None] * (
                     torch.cos(torsion)[..., None] * in_plane
                     + torch.sin(torsion)[..., None] * normal))
    return c + float(length) * direction


def _cb_from_backbone(n, ca, c):
    toward_n = (n - ca) / (n - ca).norm().clamp_min(1e-8)
    toward_c = (c - ca) / (c - ca).norm().clamp_min(1e-8)
    normal = torch.linalg.cross(toward_n, toward_c, dim=-1)
    normal = normal / normal.norm().clamp_min(1e-8)
    direction = -0.58273431 * normal + 0.56802827 * toward_n - 0.54067466 * toward_c
    return ca + BOND_CA_CB * direction / direction.norm().clamp_min(1e-8)


def build_core_from_torsions(sequence: str, phi: torch.Tensor, psi: torch.Tensor,
                             omega: torch.Tensor) -> torch.Tensor:
    """Return ``[L,7,3]`` core coordinates with exact peptide bond lengths."""
    length = len(sequence)
    if any(t.shape != (length,) for t in (phi, psi, omega)):
        raise ValueError("phi/psi/omega must each have shape [L]")
    dtype, device = phi.dtype, phi.device
    core = torch.zeros((length, 7, 3), dtype=dtype, device=device)
    core[0, ATOM_N] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    core[0, ATOM_CA] = torch.tensor([BOND_N_CA, 0.0, 0.0], dtype=dtype, device=device)
    # The root frame is canonical. Undefined terminal torsions must not affect
    # it; their state slots are masked by extract_backbone_torsions.
    initial_angle = torch.as_tensor(math.radians(180.0 - 111.2), dtype=dtype, device=device)
    core[0, ATOM_C] = core[0, ATOM_CA] + BOND_CA_C * torch.stack((torch.cos(initial_angle), torch.sin(initial_angle), initial_angle * 0))
    for index in range(1, length):
        previous = core[index - 1]
        core[index, ATOM_N] = place_atom(previous[ATOM_N], previous[ATOM_CA], previous[ATOM_C],
                                         BOND_C_N, math.radians(116.2), psi[index - 1])
        core[index, ATOM_CA] = place_atom(previous[ATOM_CA], previous[ATOM_C], core[index, ATOM_N],
                                          BOND_N_CA, math.radians(121.7), omega[index - 1])
        core[index, ATOM_C] = place_atom(previous[ATOM_C], core[index, ATOM_N], core[index, ATOM_CA],
                                         BOND_CA_C, math.radians(111.2), phi[index])
    # Carbonyl O is placed in the local peptide plane; psi belongs to the
    # following peptide bond and is not a carbonyl-O torsion.
    for index in range(length):
        core[index, ATOM_O] = place_atom(
            core[index, ATOM_N], core[index, ATOM_CA], core[index, ATOM_C],
            BOND_C_O, math.radians(120.8),
            torch.as_tensor(torch.pi, dtype=dtype, device=device),
        )
        if sequence[index] != "G":
            core[index, ATOM_CB] = _cb_from_backbone(core[index, ATOM_N], core[index, ATOM_CA], core[index, ATOM_C])
    return core


def build_core_batch(sequences: list[str], torsions: torch.Tensor,
                     token_mask: torch.Tensor | None = None) -> torch.Tensor:
    if torsions.ndim != 3 or torsions.shape[-1] != 3 or torsions.shape[0] != len(sequences):
        raise ValueError("torsions must have shape [B,L,3]")
    B, Lmax, _ = torsions.shape
    if token_mask is None:
        token_mask = torch.zeros((B, Lmax), dtype=torch.bool, device=torsions.device)
        for b, sequence in enumerate(sequences):
            token_mask[b, :len(sequence)] = True
    if token_mask.shape != (B, Lmax):
        raise ValueError("token_mask must have shape [B,L]")
    output = torsions.new_zeros((B, Lmax, 7, 3))
    for b, sequence in enumerate(sequences):
        length = len(sequence)
        if length > Lmax or int(token_mask[b, :length].sum()) != length:
            raise ValueError("token_mask does not contain a contiguous sequence prefix")
        output[b, :length] = build_core_from_torsions(
            sequence, torsions[b, :length, 0], torsions[b, :length, 1], torsions[b, :length, 2]
        )
    return output * token_mask[..., None, None].to(output.dtype)


def _dihedral(a, b, c, d):
    b0 = b - a
    b1 = c - b; b1 = b1 / b1.norm().clamp_min(1e-8)
    b2 = d - c
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    return torch.atan2((torch.linalg.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def extract_backbone_torsions(core: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if core.ndim != 3 or core.shape[1:] != (7, 3):
        raise ValueError("core must be [L,7,3]")
    L = core.shape[0]
    output = core.new_zeros((L, 3)); valid = torch.zeros((L, 3), dtype=torch.bool, device=core.device)
    available = torch.ones((L, 7), dtype=torch.bool, device=core.device) if mask is None else mask.bool()
    for i in range(L):
        if i > 0:
            raw_phi = _dihedral(core[i - 1, ATOM_C], core[i, ATOM_N], core[i, ATOM_CA], core[i, ATOM_C])
            output[i, 0] = torch.atan2(torch.sin(raw_phi - torch.pi), torch.cos(raw_phi - torch.pi))
            valid[i, 0] = bool(available[i - 1, ATOM_C] & available[i, ATOM_N] & available[i, ATOM_CA] & available[i, ATOM_C])
        if i + 1 < L:
            raw_psi = _dihedral(core[i, ATOM_N], core[i, ATOM_CA], core[i, ATOM_C], core[i + 1, ATOM_N])
            raw_omega = _dihedral(core[i, ATOM_CA], core[i, ATOM_C], core[i + 1, ATOM_N], core[i + 1, ATOM_CA])
            output[i, 1] = torch.atan2(torch.sin(raw_psi + torch.pi), torch.cos(raw_psi + torch.pi))
            output[i, 2] = torch.atan2(torch.sin(raw_omega + torch.pi), torch.cos(raw_omega + torch.pi))
            valid[i, 1] = bool(available[i, :4].all() & available[i + 1, ATOM_N])
            valid[i, 2] = bool(available[i, ATOM_CA] & available[i, ATOM_C] & available[i + 1, :2].all())
    return output, valid
