import torch

from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.model_mini import MiniCoreDiffusion
from lassodiff.peptide_prior import sample_peptide_prior
from lassodiff.sampler_mini import sample_mini
from lassodiff.sidechain_builder import build_atom14
from lassodiff.validation.strict_lasso import strict_lasso_check


def test_generator_outputs_complete_heavy_atoms():
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    prior = sample_peptide_prior(candidate, mode="single_crossing", noise_scale=0)
    atoms, mask = build_atom14(prior.coordinates, candidate)
    assert atoms.shape == (8, 14, 3)
    assert torch.isfinite(atoms[mask]).all()


def test_generated_structure_passes_bond_checker_for_programmatic_seed():
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    prior = sample_peptide_prior(candidate, mode="single_crossing", noise_scale=0)
    atoms, mask = build_atom14(prior.coordinates, candidate)
    result = strict_lasso_check(prior.coordinates, prior.atom_mask, candidate, atom14_coordinates=atoms, atom14_atom_mask=mask)
    assert result.backbone_valid


def test_strict_validator_detects_zero_one_multiple_crossings():
    candidate = CandidateCondition("LLQRNGRDRLILSKN", 7, 9)
    expected = {"no_crossing": 0, "double_crossing": 2}
    single = sample_peptide_prior(candidate, mode="single_crossing", noise_scale=0)
    assert strict_lasso_check(single.coordinates, single.atom_mask, candidate).crossing_count == 1
    for corruption, count in expected.items():
        sample = sample_peptide_prior(candidate, mode="topology_corrupted", corruption=corruption, noise_scale=0)
        assert strict_lasso_check(sample.coordinates, sample.atom_mask, candidate).crossing_count == count


def test_construction_and_screening_metrics_are_separate():
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    model = MiniCoreDiffusion(hidden_dim=24, blocks=1)
    construction = sample_mini(model, candidate, inference_mode="construction", steps=2)
    screening = sample_mini(model, candidate, inference_mode="screening", steps=2)
    assert construction.prior_mode == "single_crossing"
    assert screening.prior_mode == "open_chain"
