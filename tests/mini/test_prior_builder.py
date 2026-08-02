import torch

from lassodiff.atom_schema_lasso import ATOM_C, ATOM_CA, ATOM_N, CandidateCondition
from lassodiff.peptide_prior import sample_peptide_prior
from lassodiff.topology_adapter import CandidateBatch
from lassodiff.validation.threading_mini import hard_threading_check_ca


SEQUENCE = "LLQRNGRDRLILSKN"


def _candidate():
    return CandidateCondition(SEQUENCE, 7, 9)


def _topology(sample, candidate):
    batch = CandidateBatch(
        torch.tensor([[candidate.k]]), torch.tensor([[candidate.p]]), torch.tensor([[candidate.k]]),
        torch.ones((1, 1)), torch.ones((1, 1), dtype=torch.bool),
    )
    return hard_threading_check_ca(
        sample.coordinates[:, ATOM_CA][None, None], torch.ones((1, len(SEQUENCE)), dtype=torch.bool),
        batch, sample.atom_mask[:, ATOM_CA][None, None],
    )


def test_open_prior_has_valid_peptide_bonds():
    sample = sample_peptide_prior(_candidate(), mode="open_chain", generator=torch.Generator().manual_seed(1), noise_scale=0)
    bonds = (sample.coordinates[:-1, ATOM_C] - sample.coordinates[1:, ATOM_N]).norm(dim=-1)
    assert torch.allclose(bonds, torch.full_like(bonds, 1.329), atol=2e-4)


def test_single_crossing_prior_has_exactly_one_crossing():
    candidate = _candidate()
    sample = sample_peptide_prior(candidate, mode="single_crossing", generator=torch.Generator().manual_seed(2), noise_scale=0)
    result = _topology(sample, candidate)
    assert result.crossing_count.item() == 1
    assert result.plug_consistent.item()


def test_corrupted_prior_can_create_zero_and_multiple_crossings():
    candidate = _candidate()
    zero = sample_peptide_prior(candidate, mode="topology_corrupted", corruption="no_crossing", noise_scale=0)
    multiple = sample_peptide_prior(candidate, mode="topology_corrupted", corruption="double_crossing", noise_scale=0)
    assert _topology(zero, candidate).crossing_count.item() == 0
    assert _topology(multiple, candidate).crossing_count.item() >= 2


def test_prior_uses_no_template_coordinates(monkeypatch):
    import lassodiff.pdb_utils as pdb_utils
    monkeypatch.setattr(pdb_utils, "parse_pdb_residues", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("PDB read")))
    sample = sample_peptide_prior(_candidate(), mode="open_chain", generator=torch.Generator().manual_seed(3))
    assert torch.isfinite(sample.coordinates).all()


def test_same_sequence_different_seed_gives_different_structure():
    first = sample_peptide_prior(_candidate(), mode="open_chain", generator=torch.Generator().manual_seed(4))
    second = sample_peptide_prior(_candidate(), mode="open_chain", generator=torch.Generator().manual_seed(5))
    assert not torch.allclose(first.coordinates, second.coordinates)


def test_candidate_kp_changes_prior_geometry():
    sequence = "AAAADAEAAAAAA"
    first = CandidateCondition(sequence, 4, 7)
    second = CandidateCondition(sequence, 6, 8)
    a = sample_peptide_prior(first, mode="single_crossing", noise_scale=0).coordinates
    b = sample_peptide_prior(second, mode="single_crossing", noise_scale=0).coordinates
    assert not torch.allclose(a, b)
