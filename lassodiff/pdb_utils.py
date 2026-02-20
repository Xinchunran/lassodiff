from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import torch


AA3_TO_AA1: Dict[str, str] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "ASX": "B",
    "GLX": "Z",
    "SEC": "U",
    "PYL": "O",
}


def parse_pdb_residues(pdb_path: str) -> List[Tuple[str, Tuple[int, str], Dict[str, List[float]]]]:
    residues: Dict[Tuple[int, str], Dict[str, List[float]]] = {}
    names: Dict[Tuple[int, str], str] = {}
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            resname = line[17:20].strip()
            resseq = int(line[22:26].strip())
            icode = line[26].strip()
            key = (resseq, icode)
            atom = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if key not in residues:
                residues[key] = {}
            residues[key][atom] = [x, y, z]
            names[key] = resname
    ordered = sorted(residues.items(), key=lambda x: (x[0][0], x[0][1]))
    out = []
    for key, atoms in ordered:
        out.append((names.get(key, "UNK"), key, atoms))
    return out


def parse_pdb_sequence(pdb_path: str) -> str:
    residues = parse_pdb_residues(pdb_path)
    seq = []
    for resname, _key, _atoms in residues:
        seq.append(AA3_TO_AA1.get(resname, "X"))
    return "".join(seq)


def _normalize_acceptor_type(acceptor_type: Optional[str]) -> Optional[str]:
    if acceptor_type is None:
        return None
    t = acceptor_type.strip().upper()
    if t in ("ASP", "D"):
        return "ASP"
    if t in ("GLU", "E"):
        return "GLU"
    return None


def _resolve_acceptor_index(residues, acceptor_index: Optional[int]) -> Optional[int]:
    if acceptor_index is None:
        return None
    if acceptor_index < 0:
        return None
    for i, (_resname, (resseq, _icode), _atoms) in enumerate(residues):
        if resseq == acceptor_index:
            return i
    if 1 <= acceptor_index <= len(residues):
        return acceptor_index - 1
    return None


def parse_pdb_ca_coords(pdb_path: str) -> torch.Tensor:
    residues = parse_pdb_residues(pdb_path)
    coords = []
    for _resname, _key, atoms in residues:
        ca = atoms.get("CA")
        if ca is None:
            coords.append([0.0, 0.0, 0.0])
        else:
            coords.append(ca)
    return torch.tensor(coords, dtype=torch.float32)


def process_structure_backbone(
    pdb_path: str,
    iso_acceptor_index: Optional[int] = None,
    iso_acceptor_type: Optional[str] = None,
) -> Dict[str, torch.Tensor | str]:
    residues = parse_pdb_residues(pdb_path)
    seq = []
    coords = []
    mask = []
    residue_index = []
    acceptor_idx = _resolve_acceptor_index(residues, iso_acceptor_index)
    acceptor_type = _normalize_acceptor_type(iso_acceptor_type)
    for i, (resname, (resseq, _icode), atoms) in enumerate(residues):
        seq.append(AA3_TO_AA1.get(resname, "X"))
        residue_index.append(resseq)
        atom_coords = []
        atom_mask = []
        for atom_name in ("N", "CA", "C", "O"):
            pos = atoms.get(atom_name)
            if pos is None:
                atom_coords.append([0.0, 0.0, 0.0])
                atom_mask.append(False)
            else:
                atom_coords.append(pos)
                atom_mask.append(True)
        this_acceptor_type = acceptor_type
        if this_acceptor_type is None and acceptor_idx is not None and i == acceptor_idx:
            if resname == "ASP":
                this_acceptor_type = "ASP"
            elif resname == "GLU":
                this_acceptor_type = "GLU"
        if acceptor_idx is not None and i == acceptor_idx and this_acceptor_type in ("ASP", "GLU"):
            c_name = "CG" if this_acceptor_type == "ASP" else "CD"
            o1_name, o2_name = ("OD1", "OD2") if this_acceptor_type == "ASP" else ("OE1", "OE2")
            for atom_name in (c_name, o1_name, o2_name):
                pos = atoms.get(atom_name)
                if pos is None:
                    atom_coords.append([0.0, 0.0, 0.0])
                    atom_mask.append(False)
                else:
                    atom_coords.append(pos)
                    atom_mask.append(True)
        else:
            for _ in range(3):
                atom_coords.append([0.0, 0.0, 0.0])
                atom_mask.append(False)
        coords.append(atom_coords)
        mask.append(atom_mask)
    return {
        "seq": "".join(seq),
        "coords": torch.tensor(coords, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "residue_index": torch.tensor(residue_index, dtype=torch.long),
    }


def process_structure_ca(pdb_path: str) -> Dict[str, torch.Tensor | str]:
    residues = parse_pdb_residues(pdb_path)
    seq = []
    coords = []
    mask = []
    residue_index = []
    for resname, (resseq, _icode), atoms in residues:
        seq.append(AA3_TO_AA1.get(resname, "X"))
        ca = atoms.get("CA")
        if ca is None:
            coords.append([0.0, 0.0, 0.0])
            mask.append(False)
        else:
            coords.append(ca)
            mask.append(True)
        residue_index.append(resseq)
    return {
        "seq": "".join(seq),
        "coords": torch.tensor(coords, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "residue_index": torch.tensor(residue_index, dtype=torch.long),
    }
