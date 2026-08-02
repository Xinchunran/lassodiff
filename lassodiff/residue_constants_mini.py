"""Residue-specific Atom14 names, covalent trees, elements and chi definitions."""
from __future__ import annotations

from dataclasses import dataclass


ATOM14_NAMES = {
    "A": ("N", "CA", "C", "O", "CB"), "R": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "N": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"), "D": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "C": ("N", "CA", "C", "O", "CB", "SG"), "Q": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "E": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"), "G": ("N", "CA", "C", "O"),
    "H": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "I": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"), "L": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "K": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"), "M": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "F": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "P": ("N", "CA", "C", "O", "CB", "CG", "CD"), "S": ("N", "CA", "C", "O", "CB", "OG"),
    "T": ("N", "CA", "C", "O", "CB", "OG1", "CG2"), "W": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "Y": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "V": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
}

ELEMENTS = {name: ("N" if name.startswith("N") else "O" if name.startswith("O") else "S" if name.startswith("S") else "C")
            for names in ATOM14_NAMES.values() for name in names}
ELEMENTS.update({"N": "N", "CA": "C", "C": "C", "O": "O"})

SIDECHAIN_PARENT = {
    "A": {}, "R": {"CG": "CB", "CD": "CG", "NE": "CD", "CZ": "NE", "NH1": "CZ", "NH2": "CZ"},
    "N": {"CG": "CB", "OD1": "CG", "ND2": "CG"}, "D": {"CG": "CB", "OD1": "CG", "OD2": "CG"},
    "C": {"SG": "CB"}, "Q": {"CG": "CB", "CD": "CG", "OE1": "CD", "NE2": "CD"},
    "E": {"CG": "CB", "CD": "CG", "OE1": "CD", "OE2": "CD"}, "G": {},
    "H": {"CG": "CB", "ND1": "CG", "CD2": "CG", "CE1": "CD2", "NE2": "CE1"},
    "I": {"CG1": "CB", "CG2": "CB", "CD1": "CG1"}, "L": {"CG": "CB", "CD1": "CG", "CD2": "CG"},
    "K": {"CG": "CB", "CD": "CG", "CE": "CD", "NZ": "CE"}, "M": {"CG": "CB", "SD": "CG", "CE": "SD"},
    "F": {"CG": "CB", "CD1": "CG", "CD2": "CG", "CE1": "CD1", "CE2": "CD2", "CZ": "CE1"},
    "P": {"CG": "CB", "CD": "CG"}, "S": {"OG": "CB"}, "T": {"OG1": "CB", "CG2": "CB"},
    "W": {"CG": "CB", "CD1": "CG", "CD2": "CG", "NE1": "CD1", "CE2": "NE1", "CE3": "CD2", "CZ2": "CE2", "CZ3": "CE3", "CH2": "CZ2"},
    "Y": {"CG": "CB", "CD1": "CG", "CD2": "CG", "CE1": "CD1", "CE2": "CD2", "CZ": "CE1", "OH": "CZ"},
    "V": {"CG1": "CB", "CG2": "CB"},
}

CHI_ATOMS = {
    "R": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
    "N": (("N", "CA", "CB", "CG"),), "D": (("N", "CA", "CB", "CG"),), "Q": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "E": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")), "H": (("N", "CA", "CB", "CG"),),
    "I": (("N", "CA", "CB", "CG1"),), "L": (("N", "CA", "CB", "CG"),), "K": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")),
    "M": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD")), "P": (("N", "CA", "CB", "CG"),),
    "F": (("N", "CA", "CB", "CG"),), "W": (("N", "CA", "CB", "CG"),), "Y": (("N", "CA", "CB", "CG"),), "C": (("N", "CA", "CB", "SG"),),
}

SYMMETRIC_ATOM_PAIRS = {"D": (("OD1", "OD2"),), "E": (("OE1", "OE2"),), "F": (("CD1", "CD2"), ("CE1", "CE2")), "Y": (("CD1", "CD2"), ("CE1", "CE2")), "R": (("NH1", "NH2"),)}


def padded_atom14_names(sequence: str, acceptor_index: int | None = None) -> tuple[tuple[str, ...], ...]:
    rows = []
    for index, aa in enumerate(sequence):
        names = list(ATOM14_NAMES[aa])
        if acceptor_index is not None and index == acceptor_index:
            names.remove("OD2" if aa == "D" else "OE2")
        rows.append(tuple(names + [""] * (14 - len(names))))
    return tuple(rows)
