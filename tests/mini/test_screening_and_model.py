import inspect

import pytest
import torch

from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.candidate_viability import CandidateViabilityHead
from lassodiff.model_mini import MiniCoreDiffusion
from lassodiff.sampler_mini import sample_mini


def test_screening_disables_iso_projection():
    model = MiniCoreDiffusion(hidden_dim=24, blocks=1)
    result = sample_mini(model, CandidateCondition("AAADRAAA", 3, 5), inference_mode="screening", steps=2)
    assert not result.projection_used
    assert result.prior_mode == "open_chain"


def test_screening_disables_topology_guidance():
    model = MiniCoreDiffusion(hidden_dim=24, blocks=1)
    result = sample_mini(model, CandidateCondition("AAADRAAA", 3, 5), inference_mode="screening", steps=2)
    assert not result.topology_guidance_used
    with pytest.raises(ValueError, match="open-chain"):
        sample_mini(model, CandidateCondition("AAADRAAA", 3, 5), inference_mode="screening", prior_mode="single_crossing", steps=2)


def test_viability_head_does_not_read_generated_coordinates():
    parameters = inspect.signature(CandidateViabilityHead.forward).parameters
    assert all("coord" not in name and name not in {"x", "x_t"} for name in parameters)


def test_wrong_candidate_scores_below_correct_candidate_after_tiny_fit():
    torch.manual_seed(12)
    head = CandidateViabilityHead(hidden_dim=24, layers=1, heads=4)
    aa = torch.tensor([[0, 5, 2, 14, 0, 0, 0]])
    mask = torch.ones_like(aa, dtype=torch.bool)
    k = torch.tensor([[2, 2]])
    p = torch.tensor([[4, 5]])
    optimizer = torch.optim.Adam(head.parameters(), lr=.02)
    for _ in range(50):
        logits = head(aa, mask, k, p)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1., 0.]]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    head.eval()
    logits = head(aa, mask, k, p)
    assert logits[0, 0] > logits[0, 1]
