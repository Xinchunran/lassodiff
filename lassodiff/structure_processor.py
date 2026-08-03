"""PDB normalization for the Mini core-7 and all-heavy-atom paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .atom_schema_lasso import (
    ATOM_C, ATOM_CA, ATOM_CB, ATOM_CISO, ATOM_N, ATOM_O, ATOM_OISO,
    CandidateCondition, CORE_ATOM_NAMES,
)
from .pdb_utils import AA3_TO_AA1, parse_pdb_residues


ISO_RESIDUE_NAMES = {"ASX": "ASP_ISO", "GLX": "GLU_ISO"}


@dataclass(frozen=True)
class ProcessedLassoStructure:
    sequence: str
    residue_names: tuple[str, ...]
    core_coordinates: torch.Tensor
    core_atom_mask: torch.Tensor
    heavy_atom_coordinates: tuple[dict[str, torch.Tensor], ...]
    source: str


def canonicalize_to_root_frame(coords: torch.Tensor, n0: torch.Tensor,
                               ca0: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
    """Rigidly map coordinates into the decoder's canonical root frame."""
    ex = torch.nn.functional.normalize(ca0 - n0, dim=-1)
    c_direction = c0 - n0
    ey = c_direction - (c_direction * ex).sum(dim=-1, keepdim=True) * ex
    ey = torch.nn.functional.normalize(ey, dim=-1)
    ez = torch.linalg.cross(ex, ey, dim=-1)
    rotation = torch.stack((ex, ey, ez), dim=-1)
    return (coords - n0) @ rotation


def _sequence_letter(resname: str, *, is_acceptor: bool) -> str:
    if is_acceptor and resname == "ASX":
        return "D"
    if is_acceptor and resname == "GLX":
        return "E"
    return AA3_TO_AA1.get(resname, "X")


def process_lasso_structure(
    pdb_path: str | Path,
    candidate: CandidateCondition,
) -> ProcessedLassoStructure:
    """Read only heavy atoms and map ASX/GLX to formed-amide chemistry."""
    residues = parse_pdb_residues(str(pdb_path))
    if len(residues) != len(candidate.sequence):
        raise ValueError("PDB residue count does not match candidate sequence")
    observed = "".join(
        _sequence_letter(name, is_acceptor=index == candidate.k)
        for index, (name, _key, _atoms) in enumerate(residues)
    )
    if observed != candidate.sequence:
        raise ValueError(f"PDB sequence {observed} does not match {candidate.sequence}")

    coordinates = torch.zeros((len(residues), len(CORE_ATOM_NAMES), 3), dtype=torch.float32)
    mask = torch.zeros((len(residues), len(CORE_ATOM_NAMES)), dtype=torch.bool)
    normalized_names: list[str] = []
    heavy: list[dict[str, torch.Tensor]] = []
    for index, (resname, _key, atoms) in enumerate(residues):
        is_acceptor = index == candidate.k
        if is_acceptor:
            allowed = {"D": {"ASP", "ASH", "ASX"}, "E": {"GLU", "GLH", "GLX"}}
            if resname not in allowed[candidate.sequence[index]]:
                raise ValueError("reactive residue is not compatible with candidate acceptor")
            normalized_names.append(candidate.acceptor_type)
        else:
            normalized_names.append(resname)
        atom_map = {name: torch.tensor(value, dtype=torch.float32) for name, value in atoms.items() if not name.startswith("H")}
        heavy.append(atom_map)
        for slot, atom_name in ((ATOM_N, "N"), (ATOM_CA, "CA"), (ATOM_C, "C"), (ATOM_O, "O")):
            if atom_name in atom_map:
                coordinates[index, slot] = atom_map[atom_name]
                mask[index, slot] = True
        if "CB" in atom_map and candidate.sequence[index] != "G":
            coordinates[index, ATOM_CB] = atom_map["CB"]
            mask[index, ATOM_CB] = True
        if is_acceptor:
            carbon, oxygen = (("CG", "OD1") if candidate.sequence[index] == "D" else ("CD", "OE1"))
            if carbon not in atom_map or oxygen not in atom_map:
                raise ValueError("formed acceptor is missing CISO or its single carbonyl oxygen")
            coordinates[index, ATOM_CISO] = atom_map[carbon]
            coordinates[index, ATOM_OISO] = atom_map[oxygen]
            mask[index, ATOM_CISO] = mask[index, ATOM_OISO] = True
            removed_oxygen = "OD2" if candidate.sequence[index] == "D" else "OE2"
            if resname in ISO_RESIDUE_NAMES and removed_oxygen in atom_map:
                raise ValueError("ASX/GLX formed amide must not contain a second carboxylate oxygen")

    required = candidate.core_atom_mask
    if not bool(mask[required].all()):
        missing = [
            f"{i}:{CORE_ATOM_NAMES[a]}" for i, a in torch.nonzero(required & ~mask).tolist()
        ]
        raise ValueError(f"PDB is missing required core atoms: {', '.join(missing[:8])}")
    if not bool(torch.isfinite(coordinates[mask]).all()):
        raise ValueError("PDB contains non-finite core coordinates")
    return ProcessedLassoStructure(
        observed, tuple(normalized_names), coordinates, mask, tuple(heavy), str(Path(pdb_path)),
    )


def isopeptide_distance(structure: ProcessedLassoStructure, candidate: CandidateCondition) -> float:
    nterm = structure.core_coordinates[0, ATOM_N]
    ciso = structure.core_coordinates[candidate.k, ATOM_CISO]
    return float(torch.linalg.vector_norm(nterm - ciso))
