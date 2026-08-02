"""Write Mini Atom14 heavy atoms with formed ASX/GLX residue naming."""
from __future__ import annotations

from pathlib import Path

import torch

from .atom_schema_lasso import CandidateCondition
from .sidechain_builder import atom14_names


AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def write_atom14_pdb(path: str | Path, coordinates: torch.Tensor, atom_mask: torch.Tensor, candidate: CandidateCondition):
    if coordinates.shape != (len(candidate.sequence), 14, 3) or atom_mask.shape != coordinates.shape[:-1]:
        raise ValueError("PDB writer expects Atom14 [L,14,3]")
    names = atom14_names(candidate.sequence, candidate)
    serial, lines = 1, []
    for residue, aa in enumerate(candidate.sequence):
        resname = ("ASX" if aa == "D" else "GLX") if residue == candidate.k else AA1_TO_AA3[aa]
        for slot, atom_name in enumerate(names[residue]):
            if not atom_name or not bool(atom_mask[residue, slot]):
                continue
            x, y, z = (float(value) for value in coordinates[residue, slot])
            element = atom_name[0]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} A{residue + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
            )
            serial += 1
    lines.append("TER\nEND\n")
    Path(path).write_text("".join(lines), encoding="utf-8")
