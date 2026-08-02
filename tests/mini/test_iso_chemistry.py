from pathlib import Path

import pytest
import torch

from lassodiff.atom_schema_lasso import ATOM_CISO, ATOM_N, ATOM_OISO, CandidateCondition
from lassodiff.structure_processor import process_lasso_structure


MIN1 = Path("structure/LP_WP_069782233/min1.pdb")
SEQUENCE = "LLQRNGRDRLILSKN"


@pytest.mark.skipif(not MIN1.is_file(), reason="LassoPred min1 fixture is unavailable")
def test_asx_has_single_carbonyl_oxygen():
    candidate = CandidateCondition(SEQUENCE, 7, 9)
    structure = process_lasso_structure(MIN1, candidate)
    assert structure.residue_names[7] == "ASP_ISO"
    assert structure.core_atom_mask[7, ATOM_CISO]
    assert structure.core_atom_mask[7, ATOM_OISO]
    assert "OD2" not in structure.heavy_atom_coordinates[7]


@pytest.mark.skipif(not MIN1.is_file(), reason="LassoPred min1 fixture is unavailable")
def test_iso_edge_is_nterm_n_to_acceptor_ciso():
    candidate = CandidateCondition(SEQUENCE, 7, 9)
    structure = process_lasso_structure(MIN1, candidate)
    expected = (structure.core_coordinates[0, ATOM_N] - structure.core_coordinates[7, ATOM_CISO]).norm()
    assert 1.2 <= float(expected) <= 1.7
    assert any(edge.bond_type == "isopeptide" and edge.left_residue == 0 and edge.right_residue == 7 for edge in candidate.covalent_edges)


@pytest.mark.skipif(not MIN1.is_file(), reason="LassoPred min1 fixture is unavailable")
def test_iso_bond_geometry_matches_dataset_distribution():
    structure = process_lasso_structure(MIN1, CandidateCondition(SEQUENCE, 7, 9))
    distance = torch.linalg.vector_norm(structure.core_coordinates[0, ATOM_N] - structure.core_coordinates[7, ATOM_CISO])
    assert float(distance) == pytest.approx(1.558, abs=.02)


def test_invalid_acceptor_fails_closed():
    with pytest.raises(ValueError, match="Asp or Glu"):
        CandidateCondition("AAAAAA", 2, 4)
    with pytest.raises(ValueError, match="too short"):
        CandidateCondition("AADAAA", 2, 5)
