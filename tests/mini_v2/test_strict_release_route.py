import torch

from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.evaluation_mini_v2 import evaluate_generated_candidate
from lassodiff.peptide_prior import sample_peptide_prior


def test_release_sample_is_always_decided_by_strict_checker():
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    prior = sample_peptide_prior(candidate, mode="open_chain", noise_scale=0)
    row = evaluate_generated_candidate(prior.coordinates, prior.atom_mask, candidate)
    assert set(("finite", "strict_valid", "crossing_count", "rejection_reasons")) <= set(row)
    assert row["finite"] is True
    # The open-chain prior is allowed to be invalid; validity is not inferred from finite coordinates.
    assert row["strict_valid"] is False
