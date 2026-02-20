from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import torch

from .pdb_utils import process_structure_ca


def _read_fasta(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return None
    if lines[0].startswith(">"):
        lines = lines[1:]
    return "".join(lines).strip()


def process_structure(pdb_path: str) -> Dict[str, torch.Tensor | str]:
    out = process_structure_ca(pdb_path)
    pdb = Path(pdb_path)
    fasta_path = pdb.with_suffix(".fasta")
    if not fasta_path.exists():
        fasta_path = pdb.parent / f"{pdb.parent.name}.fasta"
    seq = _read_fasta(fasta_path)
    if seq and len(seq) == len(out["seq"]):
        out["seq"] = seq
    return out
