from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.model_v3 import LassoDiffOpenDDEV3
from lassodiff.opendde_bridge.reasoner import FrozenOpenDDEReasoner
from lassodiff.opendde_bridge.schema import OpenDDEReasoningState
from lassodiff.sequence_gate import SequenceAssessment
from lassodiff.topology_adapter import CandidateBatch


class FrozenStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


def _state(pair_scale=1.0):
    torch.manual_seed(9)
    B, L = 2, 6
    return OpenDDEReasoningState(
        single=torch.randn(B, L, 12), pair=torch.randn(B, L, L, 10) * pair_scale,
        token_mask=torch.ones(B, L, dtype=torch.bool), residue_index=torch.arange(L)[None].expand(B, -1),
        sequence_hashes=("a", "b"), checkpoint_sha256="hash", opendde_commit="commit",
        feature_schema_version=1,
    )


def _candidates():
    return CandidateBatch(
        k=torch.tensor([[1, 1], [1, 1]]), p=torch.tensor([[3, 4], [3, 4]]),
        acceptor_index=torch.tensor([[2, 2], [2, 2]]), prior=torch.tensor([[0.8, 0.2], [0.3, 0.7]]),
        candidate_mask=torch.ones(2, 2, dtype=torch.bool),
    )


def _model():
    stub = FrozenStub()
    reasoner = FrozenOpenDDEReasoner(stub, lambda _: _state())
    return LassoDiffOpenDDEV3(reasoner, 12, 10, c_s=16, c_z=16, c_a=24, n_heads=4, diffusion_blocks=2)


def test_structure_output_depends_on_opendde_pair_state():
    model = _model().eval()
    candidates = _candidates()
    x = torch.randn(2, 2, 6, 7, 3)
    mask = torch.ones(2, 2, 6, 7, dtype=torch.bool)
    t = torch.rand(2)
    real = model.predict_velocity(_state(1.0), candidates, x, t, mask).velocity
    zero = model.predict_velocity(_state(0.0), candidates, x, t, mask).velocity
    relative = (real - zero).norm() / real.norm().clamp_min(1e-8)
    assert relative > 1e-3


def test_candidate_permutation_equivariance():
    model = _model().eval()
    candidates = _candidates()
    x = torch.randn(2, 2, 6, 7, 3)
    mask = torch.ones(2, 2, 6, 7, dtype=torch.bool)
    t = torch.rand(2)
    output = model.predict_velocity(_state(), candidates, x, t, mask).velocity
    perm = torch.tensor([1, 0])
    permuted = CandidateBatch(
        candidates.k[:, perm], candidates.p[:, perm], candidates.acceptor_index[:, perm],
        candidates.prior[:, perm], candidates.candidate_mask[:, perm],
    )
    changed = model.predict_velocity(_state(), permuted, x[:, perm], t, mask[:, perm]).velocity
    torch.testing.assert_close(output[:, perm], changed, atol=2e-5, rtol=2e-5)


def test_structure_prior_is_detached_and_generator_does_not_update_gate():
    model = _model()
    assessment = model.assess_sequence(_state())
    prior = model.structure_prior(assessment, _candidates().candidate_mask)
    assert not prior.requires_grad
    x = torch.randn(2, 2, 6, 7, 3)
    mask = torch.ones(2, 2, 6, 7, dtype=torch.bool)
    output = model.predict_velocity(_state(), _candidates(), x, torch.rand(2), mask)
    output.velocity.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.sequence_gate.parameters())
    assert all(parameter.grad is None for parameter in model.reasoner.parameters())
    for module in (model.reasoning_adapter, model.topology_adapter, model.structural_tokens, model.diffusion):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())


def test_negative_screening_never_calls_diffusion(monkeypatch):
    model = _model().eval()
    monkeypatch.setattr(model.diffusion, "forward", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("diffusion called")))
    result = model.screen_state(
        _state(), reject_threshold=0.99, accept_threshold=0.999, ood_threshold=0.999
    )
    assert len(result["decision"]) == 2
    assert result["projection_used"] is False and result["topology_guidance_used"] is False
