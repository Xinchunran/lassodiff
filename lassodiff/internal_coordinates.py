"""Template-free peptide geometry builders used by Mini priors and sidechains."""
from __future__ import annotations

import math

import torch

from .atom_schema_lasso import (
    ATOM_C, ATOM_CA, ATOM_CB, ATOM_CISO, ATOM_N, ATOM_O, ATOM_OISO,
    CandidateCondition,
)


BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329
BOND_C_O = 1.231
BOND_CA_CB = 1.522
CA_TRACE_SPACING = 3.80


def place_atom(a, b, c, length: float, angle_degrees: float, dihedral_degrees: float):
    """Natural-extension-reference-frame placement of atom d after a-b-c."""
    angle = torch.as_tensor(math.radians(angle_degrees), dtype=c.dtype, device=c.device)
    dihedral = torch.as_tensor(math.radians(dihedral_degrees), dtype=c.dtype, device=c.device)
    bc = c - b
    bc = bc / bc.norm().clamp_min(1e-8)
    normal = torch.linalg.cross(b - a, bc, dim=-1)
    if float(normal.norm()) < 1e-7:
        fallback = torch.tensor([0.0, 0.0, 1.0], dtype=c.dtype, device=c.device)
        if abs(float(torch.dot(fallback, bc))) > 0.9:
            fallback = torch.tensor([0.0, 1.0, 0.0], dtype=c.dtype, device=c.device)
        normal = torch.linalg.cross(fallback, bc, dim=-1)
    normal = normal / normal.norm().clamp_min(1e-8)
    in_plane = torch.linalg.cross(normal, bc, dim=-1)
    direction = (
        -torch.cos(angle) * bc
        + torch.sin(angle) * (torch.cos(dihedral) * in_plane + torch.sin(dihedral) * normal)
    )
    return c + float(length) * direction


def _cb_from_backbone(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    toward_n = (n - ca) / (n - ca).norm().clamp_min(1e-8)
    toward_c = (c - ca) / (c - ca).norm().clamp_min(1e-8)
    normal = torch.linalg.cross(toward_n, toward_c, dim=-1)
    normal = normal / normal.norm().clamp_min(1e-8)
    direction = -0.58273431 * normal + 0.56802827 * toward_n - 0.54067466 * toward_c
    return ca + BOND_CA_CB * direction / direction.norm().clamp_min(1e-8)


def build_backbone_from_torsions(
    sequence: str,
    phi: torch.Tensor,
    psi: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Build N/CA/C/O/CB from bond geometry; no structure coordinates are read."""
    length = len(sequence)
    if any(tensor.shape != (length,) for tensor in (phi, psi, omega)):
        raise ValueError("phi/psi/omega must each have shape [L]")
    dtype, device = phi.dtype, phi.device
    core = torch.zeros((length, 7, 3), dtype=dtype, device=device)
    core[0, ATOM_N] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    core[0, ATOM_CA] = torch.tensor([BOND_N_CA, 0.0, 0.0], dtype=dtype, device=device)
    theta = math.radians(180.0 - 111.2)
    core[0, ATOM_C] = core[0, ATOM_CA] + BOND_CA_C * torch.tensor(
        [math.cos(theta), math.sin(theta), 0.0], dtype=dtype, device=device,
    )
    for index in range(1, length):
        previous = core[index - 1]
        core[index, ATOM_N] = place_atom(
            previous[ATOM_N], previous[ATOM_CA], previous[ATOM_C],
            BOND_C_N, 116.2, float(psi[index - 1]),
        )
        core[index, ATOM_CA] = place_atom(
            previous[ATOM_CA], previous[ATOM_C], core[index, ATOM_N],
            BOND_N_CA, 121.7, float(omega[index - 1]),
        )
        core[index, ATOM_C] = place_atom(
            previous[ATOM_C], core[index, ATOM_N], core[index, ATOM_CA],
            BOND_CA_C, 111.2, float(phi[index]),
        )
    for index in range(length):
        core[index, ATOM_O] = place_atom(
            core[index, ATOM_N], core[index, ATOM_CA], core[index, ATOM_C],
            BOND_C_O, 120.8, 180.0,
        )
        if sequence[index] != "G":
            core[index, ATOM_CB] = _cb_from_backbone(
                core[index, ATOM_N], core[index, ATOM_CA], core[index, ATOM_C],
            )
    return core


def build_backbone_from_ca_trace(sequence: str, ca: torch.Tensor) -> torch.Tensor:
    """Reconstruct exact N-CA, CA-C and peptide C-N lengths around a CA trace."""
    if ca.shape != (len(sequence), 3):
        raise ValueError("CA trace shape must be [L,3]")
    length = len(sequence)
    core = torch.zeros((length, 7, 3), dtype=ca.dtype, device=ca.device)
    core[:, ATOM_CA] = ca
    global_z = torch.tensor([0.0, 0.0, 1.0], dtype=ca.dtype, device=ca.device)
    first_direction = (ca[1] - ca[0]) / (ca[1] - ca[0]).norm().clamp_min(1e-8)
    first_normal = torch.linalg.cross(first_direction, global_z, dim=-1)
    if float(first_normal.norm()) < 1e-6:
        first_normal = torch.tensor([0.0, 1.0, 0.0], dtype=ca.dtype, device=ca.device)
    first_normal = first_normal / first_normal.norm().clamp_min(1e-8)
    core[0, ATOM_N] = ca[0] - 0.85 * BOND_N_CA * first_direction + math.sqrt(1 - .85**2) * BOND_N_CA * first_normal

    for index in range(length - 1):
        direction = ca[index + 1] - ca[index]
        distance = direction.norm()
        if not 2.4 <= float(distance) <= 4.5:
            raise ValueError("CA trace step is outside reconstructable peptide geometry")
        unit = direction / distance
        normal = torch.linalg.cross(unit, global_z, dim=-1)
        if float(normal.norm()) < 1e-6:
            normal = first_normal
        normal = normal / normal.norm().clamp_min(1e-8)
        # Choose the CA->C projection so that the next N can lie on both the
        # C--N and N--CA spheres.  A fixed projection is only valid near a
        # 3.8 A CA trace and incorrectly rejects still-realizable ~4.1 A
        # procedural steps.
        preferred_cosine = .82
        preferred_separation2 = (
            float(distance) ** 2 + BOND_CA_C ** 2
            - 2 * float(distance) * BOND_CA_C * preferred_cosine
        )
        preferred_separation = math.sqrt(max(preferred_separation2, 0.0))
        lower = abs(BOND_C_N - BOND_N_CA) + 1e-3
        upper = BOND_C_N + BOND_N_CA - 1e-3
        target_separation = min(max(preferred_separation, lower), upper)
        cosine = (
            float(distance) ** 2 + BOND_CA_C ** 2 - target_separation ** 2
        ) / (2 * float(distance) * BOND_CA_C)
        cosine = min(max(cosine, -1.0), 1.0)
        c_direction = cosine * unit + math.sqrt(max(1 - cosine**2, 0.0)) * normal
        c_atom = ca[index] + BOND_CA_C * c_direction
        to_next_ca = ca[index + 1] - c_atom
        separation = to_next_ca.norm()
        if not abs(BOND_C_N - BOND_N_CA) < float(separation) < BOND_C_N + BOND_N_CA:
            raise ValueError("CA trace cannot realize peptide C-N bond")
        axis = to_next_ca / separation
        x = (BOND_C_N**2 - BOND_N_CA**2 + float(separation)**2) / (2.0 * float(separation))
        height = math.sqrt(max(BOND_C_N**2 - x**2, 0.0))
        circle_normal = torch.linalg.cross(axis, normal, dim=-1)
        if float(circle_normal.norm()) < 1e-6:
            circle_normal = first_normal
        circle_normal = circle_normal / circle_normal.norm().clamp_min(1e-8)
        core[index, ATOM_C] = c_atom
        core[index + 1, ATOM_N] = c_atom + x * axis + height * circle_normal
    last_direction = (ca[-1] - ca[-2]) / (ca[-1] - ca[-2]).norm().clamp_min(1e-8)
    last_normal = torch.linalg.cross(last_direction, global_z, dim=-1)
    if float(last_normal.norm()) < 1e-6:
        last_normal = first_normal
    last_normal = last_normal / last_normal.norm().clamp_min(1e-8)
    final_direction = 0.82 * last_direction + math.sqrt(1 - .82**2) * last_normal
    core[-1, ATOM_C] = ca[-1] + BOND_CA_C * final_direction
    for index in range(length):
        core[index, ATOM_O] = place_atom(
            core[index, ATOM_N], core[index, ATOM_CA], core[index, ATOM_C], BOND_C_O, 120.8, 180.0,
        )
        if sequence[index] != "G":
            core[index, ATOM_CB] = _cb_from_backbone(core[index, ATOM_N], ca[index], core[index, ATOM_C])
    return core


def place_acceptor_reactive_atoms(core: torch.Tensor, candidate: CandidateCondition, *, formed: bool) -> torch.Tensor:
    """Place Asp CG/OD1 or Glu CD/OE1, optionally near formed N--C geometry."""
    output = core.clone()
    k = candidate.k
    n, ca, cb = output[k, ATOM_N], output[k, ATOM_CA], output[k, ATOM_CB]
    if candidate.sequence[k] == "D":
        ciso = place_atom(n, ca, cb, 1.52, 113.0, -60.0)
    else:
        cg = place_atom(n, ca, cb, 1.52, 113.0, -60.0)
        ciso = place_atom(ca, cb, cg, 1.52, 113.0, 180.0)
    if formed:
        nterm = output[0, ATOM_N]
        direction = ciso - nterm
        if float(direction.norm()) < 1e-6:
            direction = torch.tensor([1.0, 0.0, 0.0], dtype=core.dtype, device=core.device)
        ciso = nterm + 1.49 * direction / direction.norm()
    side = cb if candidate.sequence[k] == "D" else cg
    oiso = place_atom(ca, side, ciso, 1.24, 120.0, 180.0)
    output[k, ATOM_CISO] = ciso
    output[k, ATOM_OISO] = oiso
    return output
