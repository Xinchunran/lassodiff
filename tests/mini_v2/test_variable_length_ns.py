import torch
from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.model_mini_v2 import MiniTorsionDiffusion
from lassodiff.sampler_mini_v2 import open_chain_torsion_prior
from lassodiff.training_mini_v2 import MiniTrainingSystemV2


def test_model_supports_variable_length_and_ns():
    candidates = [CandidateCondition("AAADRAAA", 3, 5), CandidateCondition("AAAEDRAAAAA", 3, 7)]
    L = 11; aa = torch.full((2, L), 20, dtype=torch.long); mask = torch.zeros((2, L), dtype=torch.bool)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for b, c in enumerate(candidates):
        aa[b, :len(c.sequence)] = torch.tensor([alphabet.index(x) for x in c.sequence]); mask[b, :len(c.sequence)] = True
    system = MiniTrainingSystemV2.tiny_for_test(); conditioning = system.conditioner(sequences=[c.sequence for c in candidates], aa_ids=aa, token_mask=mask, k=torch.tensor([3, 3]), p=torch.tensor([5, 7]))
    state = open_chain_torsion_prior(candidates, num_samples=3, generator=torch.Generator().manual_seed(2))
    state.backbone[:, 1] += .4; state.backbone[:, 2] -= .5
    output = system.backbone(state_t=state, time=torch.zeros(2, 3), conditioning=conditioning, token_mask=mask, candidates=candidates)
    assert output.velocity.backbone.shape == (2, 3, 11, 3)
    assert torch.all(output.velocity.backbone[0, :, 8:] == 0)
    assert not torch.allclose(output.velocity.backbone[:, 0], output.velocity.backbone[:, 1])
