"""Adapter for the checked-in AlphaFold-style rigid-group constants.

The repository already carries the residue constants from the SimpleFold
implementation under ``ml-simplefold``.  This module consumes those constants
directly instead of maintaining a second, approximate tetrahedral table.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch


_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ml-simplefold/src/simplefold/utils/residue_constants.py"
)
_SPEC = importlib.util.spec_from_file_location("lassodiff_af2_residue_constants", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load AlphaFold residue constants from {_SOURCE}")
_RC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RC)

RESTYPE_INDEX = {letter: index for index, letter in enumerate(_RC.restypes)}
RESTYPE_3 = {letter: _RC.restype_1to3[letter] for letter in _RC.restypes}


def _homogeneous(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    result = torch.eye(4, dtype=rotation.dtype, device=rotation.device)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _rotation_x(angle: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=angle.dtype, device=angle.device)
    one = torch.ones((), dtype=angle.dtype, device=angle.device)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((
        torch.stack((one, zero, zero)),
        torch.stack((zero, c, -s)),
        torch.stack((zero, s, c)),
    ))


def _backbone_frame(core: torch.Tensor) -> torch.Tensor:
    """Map AF2's CA-rooted canonical coordinates to the supplied backbone."""
    n, ca, c = core[0], core[1], core[2]
    ex = torch.nn.functional.normalize(c - ca, dim=-1)
    ey = n - ca - (n - ca).dot(ex) * ex
    ey = torch.nn.functional.normalize(ey, dim=-1)
    ez = torch.linalg.cross(ex, ey, dim=-1)
    return _homogeneous(torch.stack((ex, ey, ez), dim=-1), ca)


def _chi1_frame(core: torch.Tensor) -> torch.Tensor:
    """Exact chi1 axis frame, anchored to the decoder's CB."""
    n, ca, cb = core[0], core[1], core[4]
    ex = torch.nn.functional.normalize(cb - ca, dim=-1)
    ey = n - ca - (n - ca).dot(ex) * ex
    ey = torch.nn.functional.normalize(ey, dim=-1)
    ez = torch.linalg.cross(ex, ey, dim=-1)
    return _homogeneous(torch.stack((ex, ey, ez), dim=-1), cb)


def build_atom14_from_af2_rigid_groups(
    backbone_core: torch.Tensor,
    sequence: str,
    chi_angles: torch.Tensor,
    chi_masks: torch.Tensor,
    candidate=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one residue chain from AF2/OpenFold rigid-group constants.

    The sidechain atoms are placed by composing the canonical chi frames and
    rotating each frame about its x-axis.  Backbone and lasso-reactive atoms
    remain owned by the exact core decoder; this avoids a second backbone gauge.
    """
    if backbone_core.ndim != 3 or backbone_core.shape[0] != len(sequence):
        raise ValueError("backbone_core must be [L,7,3]")
    if chi_angles.shape != (len(sequence), 4) or chi_masks.shape != (len(sequence), 4):
        raise ValueError("chi tensors must be [L,4]")
    length = len(sequence)
    output = backbone_core.new_zeros((length, 14, 3))
    mask = torch.zeros((length, 14), dtype=torch.bool, device=backbone_core.device)
    for residue, aa in enumerate(sequence):
        restype = RESTYPE_INDEX[aa]
        resname = RESTYPE_3[aa]
        names = list(_RC.restype_name_to_atom14_names[resname])
        if candidate is not None and residue == candidate.k:
            if aa == "D":
                names[names.index("OD2")] = ""
            elif aa == "E":
                names[names.index("OE2")] = ""
        base = _backbone_frame(backbone_core[residue])
        default = torch.as_tensor(
            _RC.restype_rigid_group_default_frame[restype],
            dtype=backbone_core.dtype,
            device=backbone_core.device,
        )
        frames = [base]
        frames.extend([base @ default[group] for group in (1, 2, 3)])
        for chi_index in range(4):
            if float(_RC.chi_angles_mask[restype][chi_index]) == 0.0:
                frames.append(frames[-1])
                continue
            angle = chi_angles[residue, chi_index] if chi_masks[residue, chi_index] else chi_angles.new_zeros(())
            if chi_index == 0:
                parent = _chi1_frame(backbone_core[residue])
                frames.append(parent @ _homogeneous(_rotation_x(torch.pi - angle), angle.new_zeros(3)))
            else:
                parent = frames[3 + chi_index]
                frames.append(parent @ default[4 + chi_index] @ _homogeneous(_rotation_x(torch.pi - angle), angle.new_zeros(3)))
        positions = torch.as_tensor(
            _RC.restype_atom14_rigid_group_positions[restype],
            dtype=backbone_core.dtype,
            device=backbone_core.device,
        )
        groups = _RC.restype_atom14_to_rigid_group[restype]
        for slot, name in enumerate(names):
            if not name:
                continue
            group = int(groups[slot])
            homogeneous = torch.cat((positions[slot], positions.new_ones(1)))
            output[residue, slot] = (frames[group] @ homogeneous)[:3]
            mask[residue, slot] = True
        # Exact chain kinematics owns these five atoms.  The constants are
        # used for every sidechain atom and for default O placement only.
        names_to_slot = {name: slot for slot, name in enumerate(names) if name}
        for atom, core_slot in (("N", 0), ("CA", 1), ("C", 2), ("O", 3), ("CB", 4)):
            if atom in names_to_slot and bool(backbone_core[residue, core_slot].abs().sum() > 0):
                output[residue, names_to_slot[atom]] = backbone_core[residue, core_slot]
        if candidate is not None and residue == candidate.k:
            if aa == "D":
                output[residue, names_to_slot["CG"]] = backbone_core[residue, 5]
                output[residue, names_to_slot["OD1"]] = backbone_core[residue, 6]
            elif aa == "E":
                output[residue, names_to_slot["CD"]] = backbone_core[residue, 5]
                output[residue, names_to_slot["OE1"]] = backbone_core[residue, 6]
    return output, mask


@lru_cache(maxsize=None)
def reference_bond_length(aa: str, left: str, right: str) -> float:
    """Return a bond length from the AF2 rigid-group geometry."""
    resname = RESTYPE_3[aa]
    atoms = {name: pos for name, group, pos in _RC.rigid_group_atom_positions[resname]}
    if left not in atoms or right not in atoms:
        raise KeyError((aa, left, right))
    core = torch.zeros((1, 7, 3), dtype=torch.float32)
    for name, slot in (("N", 0), ("CA", 1), ("C", 2), ("CB", 4)):
        core[0, slot] = torch.as_tensor(atoms[name])
    names = list(_RC.restype_name_to_atom14_names[resname])
    slots = {name: index for index, name in enumerate(names) if name}
    built, _ = build_atom14_from_af2_rigid_groups(
        core, aa, torch.zeros((1, 4)), torch.ones((1, 4), dtype=torch.bool), None,
    )
    return float((built[0, slots[left]] - built[0, slots[right]]).norm())
