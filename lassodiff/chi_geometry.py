"""Target-only chi extraction and residue-specific rigid-group Atom14 building."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .atom_schema_lasso import ATOM_C, ATOM_CA, ATOM_CB, ATOM_N, ATOM_O
from .af2_rigid_group import build_atom14_from_af2_rigid_groups
from .residue_constants_mini import CHI_ATOMS, SIDECHAIN_PARENT, SYMMETRIC_ATOM_PAIRS, padded_atom14_names


@dataclass
class ChiTargets:
    angles: torch.Tensor
    masks: torch.Tensor
    sin_cos: torch.Tensor
    rotamer_classes: torch.Tensor


def _place(a, b, c, length, angle, dihedral):
    bc = c - b; bc = bc / bc.norm().clamp_min(1e-8)
    normal = torch.linalg.cross(b - a, bc, dim=-1)
    z_axis = torch.tensor([0., 0., 1.], dtype=c.dtype, device=c.device)
    y_axis = torch.tensor([0., 1., 0.], dtype=c.dtype, device=c.device)
    fallback = torch.where(bc[..., 2:3].abs() > .9, y_axis, z_axis)
    fallback_normal = torch.linalg.cross(fallback, bc, dim=-1)
    normal = torch.where(normal.norm(dim=-1, keepdim=True) < 1e-7, fallback_normal, normal)
    normal = normal / normal.norm().clamp_min(1e-8)
    in_plane = torch.linalg.cross(normal, bc, dim=-1)
    return c + length * (-math.cos(angle) * bc + math.sin(angle) * (torch.cos(dihedral) * in_plane + torch.sin(dihedral) * normal))


def _dihedral(a, b, c, d):
    b0 = b - a
    b1 = c - b; b1 = b1 / b1.norm().clamp_min(1e-8)
    b2 = d - c
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    return torch.atan2((torch.linalg.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def _chi_index(aa: str, atom: str) -> int | None:
    for index, quadruplet in enumerate(CHI_ATOMS.get(aa, ())):
        if quadruplet[-1] == atom:
            return index
    return None


def _build_one(core, aa: str, residue: int, names, chi, chi_mask):
    output = core.new_zeros((14, 3))
    lookup = {name: slot for slot, name in enumerate(names) if name}
    for atom, slot in (("N", ATOM_N), ("CA", ATOM_CA), ("C", ATOM_C), ("O", ATOM_O), ("CB", ATOM_CB)):
        if atom in lookup:
            output[lookup[atom]] = core[slot]
    if aa == "G":
        return output
    # Build atoms in Atom14 order, with the exact target chi used for the
    # corresponding terminal atom in each chi quadruplet.
    built = set(("N", "CA", "C", "O", "CB"))
    for atom in names[5:]:
        if not atom:
            continue
        parent = SIDECHAIN_PARENT[aa].get(atom, "CB")
        parent_parent = SIDECHAIN_PARENT[aa].get(parent, "CA")
        if parent not in built:
            parent = "CB"
        if parent_parent not in built:
            parent_parent = "CA"
        index = _chi_index(aa, atom)
        if index is not None:
            qa, qb, qc, _ = CHI_ATOMS[aa][index]
            a, b, c = output[lookup[qa]], output[lookup[qb]], output[lookup[qc]]
        else:
            a_name = SIDECHAIN_PARENT[aa].get(parent_parent, "N")
            if a_name not in built:
                a_name = "N"
            a, b, c = output[lookup[a_name]], output[lookup[parent_parent]], output[lookup[parent]]
        angle = chi[index] if index is not None and bool(chi_mask[index]) else core.new_tensor(0.0)
        placement_dihedral = torch.atan2(torch.sin(core.new_tensor(math.pi) - angle), torch.cos(core.new_tensor(math.pi) - angle))
        output[lookup[atom]] = _place(a, b, c, 1.52, math.radians(109.5), placement_dihedral)
        built.add(atom)
    return output


def build_atom14_from_rigid_groups(backbone_core: torch.Tensor, aa_ids: torch.Tensor,
                                   chi_angles: torch.Tensor, chi_masks: torch.Tensor,
                                   candidates) -> tuple[torch.Tensor, torch.Tensor]:
    if backbone_core.ndim == 3:
        backbone_core = backbone_core[None]
        aa_ids = aa_ids[None] if aa_ids.ndim == 1 else aa_ids
        chi_angles = chi_angles[None]
        chi_masks = chi_masks[None]
        squeeze = True
    else:
        squeeze = False
    B, L = aa_ids.shape
    outputs, masks = [], []
    candidate_list = list(candidates) if isinstance(candidates, (list, tuple)) else [candidates]
    for batch, candidate in enumerate(candidate_list):
        rows, mask = build_atom14_from_af2_rigid_groups(
            backbone_core[batch], candidate.sequence,
            chi_angles[batch], chi_masks[batch], candidate,
        )
        # Unit and prior decoders may provide a core without reactive slots;
        # in that case construct only the missing reactive group from chi.
        # Production fitted cores always carry these slots and are not
        # overwritten.
        if backbone_core[batch, candidate.k, 5].norm() < 1e-6:
            reactive_cg, reactive_ciso, reactive_oiso = _build_acceptor_reactive_group(
                sequence=candidate.sequence,
                core=backbone_core[batch],
                acceptor_index=candidate.k,
                chi=chi_angles[batch, candidate.k],
                chi_mask=chi_masks[batch, candidate.k],
            )
            lookup = {name: slot for slot, name in enumerate(
                padded_atom14_names(candidate.sequence, candidate.k)[candidate.k]
            ) if name}
            if candidate.sequence[candidate.k] == "D":
                rows[candidate.k, lookup["CG"]] = reactive_ciso
                rows[candidate.k, lookup["OD1"]] = reactive_oiso
            else:
                rows[candidate.k, lookup["CG"]] = reactive_cg
                rows[candidate.k, lookup["CD"]] = reactive_ciso
                rows[candidate.k, lookup["OE1"]] = reactive_oiso
        outputs.append(rows)
        masks.append(mask)
    result = torch.stack(outputs), torch.stack(masks)
    return (result[0][0], result[1][0]) if squeeze else result


def _build_acceptor_reactive_group(*, sequence: str, core: torch.Tensor,
                                   acceptor_index: int, chi: torch.Tensor,
                                   chi_mask: torch.Tensor):
    """Build the formed acceptor carbon and oxygen from acceptor chi.

    This is deliberately a residue-specific internal-coordinate route.  It
    does not use the legacy radial sidechain fallback, so a predicted chi
    change necessarily changes the reactive atoms used by the lasso decoder.
    """
    aa = sequence[acceptor_index]
    if aa not in {"D", "E"}:
        raise ValueError("formed acceptor must be Asp or Glu")
    if core.ndim != 3 or core.shape[1:] != (7, 3):
        raise ValueError("core must be [L,7,3]")
    if chi.shape[-1] != 4 or chi_mask.shape[-1] != 4:
        raise ValueError("acceptor chi must have four padded slots")
    # Use the same AF2 rigid-group constants as the Atom14 builder.  Keeping a
    # second internal-coordinate implementation here would make CISO/CD drift
    # away from the CG produced by the full-atom route.
    all_chi = chi.new_zeros((len(sequence), 4))
    all_mask = torch.zeros_like(all_chi, dtype=torch.bool)
    all_chi[acceptor_index] = chi
    all_mask[acceptor_index] = chi_mask
    built, _ = build_atom14_from_af2_rigid_groups(
        core, sequence, all_chi, all_mask, candidate=None,
    )
    names = padded_atom14_names(sequence)[acceptor_index]
    lookup = {name: slot for slot, name in enumerate(names) if name}
    cg = built[acceptor_index, lookup["CG"]]
    if aa == "D":
        ciso = cg
        oiso = built[acceptor_index, lookup["OD1"]]
    else:
        ciso = built[acceptor_index, lookup["CD"]]
        oiso = built[acceptor_index, lookup["OE1"]]
    return cg, ciso, oiso


def build_acceptor_reactive_atoms(*, sequence: str, core: torch.Tensor,
                                  acceptor_index: int, chi: torch.Tensor,
                                  chi_mask: torch.Tensor, n_terminal_position=None):
    """Return CISO/OISO from the residue-specific formed acceptor group."""
    _cg, ciso, oiso = _build_acceptor_reactive_group(
        sequence=sequence,
        core=core,
        acceptor_index=acceptor_index,
        chi=chi,
        chi_mask=chi_mask,
    )
    return ciso, oiso


def extract_chi_angles(atom14_coordinates: torch.Tensor, atom14_mask: torch.Tensor, aa_ids: torch.Tensor, candidates) -> ChiTargets:
    if atom14_coordinates.ndim != 5:
        raise ValueError("chi extraction expects [B,M,L,14,3]")
    B, M, L = atom14_coordinates.shape[:3]
    angles = atom14_coordinates.new_zeros((B, M, L, 4)); masks = torch.zeros((B, M, L, 4), dtype=torch.bool, device=atom14_coordinates.device)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for batch, candidate in enumerate(candidates):
        names = padded_atom14_names(candidate.sequence, candidate.k)
        for residue, aa in enumerate(candidate.sequence):
            lookup = {name: slot for slot, name in enumerate(names[residue]) if name}
            for index, quadruplet in enumerate(CHI_ATOMS.get(aa, ())):
                if all(atom in lookup for atom in quadruplet):
                    slots = [lookup[atom] for atom in quadruplet]
                    valid = atom14_mask[batch, :, residue, slots].all(-1)
                    value = -_dihedral(atom14_coordinates[batch, :, residue, slots[0]], atom14_coordinates[batch, :, residue, slots[1]],
                                       atom14_coordinates[batch, :, residue, slots[2]], atom14_coordinates[batch, :, residue, slots[3]])
                    angles[batch, :, residue, index] = value
                    masks[batch, :, residue, index] = valid
    sin_cos = torch.stack((torch.sin(angles), torch.cos(angles)), -1)
    return ChiTargets(angles, masks, sin_cos, rotamer_classes_from_target_chi(angles, masks))


def rotamer_classes_from_target_chi(target_chi: torch.Tensor, chi_mask: torch.Tensor) -> torch.Tensor:
    wrapped = torch.atan2(torch.sin(target_chi), torch.cos(target_chi))
    # Labels use only target angles and are intentionally not prediction-dependent.
    return torch.remainder(torch.round((wrapped + math.pi) / (2 * math.pi) * 3), 3).long() * chi_mask.long()
