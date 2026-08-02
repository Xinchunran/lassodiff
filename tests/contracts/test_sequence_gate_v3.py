from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from lassodiff.reasoning_adapter import OpenDDEReasoningAdapter
from lassodiff.sequence_gate import LassoSequenceGate, decide_sequence


FORBIDDEN = {
    "k", "p", "lasso_feats", "is_ring", "is_loop", "is_tail",
    "acceptor_index", "candidate_rank", "closure_edge",
}


def _inputs():
    torch.manual_seed(5)
    return torch.randn(2, 7, 12), torch.randn(2, 7, 7, 10), torch.ones(2, 7, dtype=torch.bool)


def test_gate_signature_has_no_forbidden_inputs():
    parameters = set(inspect.signature(LassoSequenceGate.forward).parameters)
    assert not (parameters & FORBIDDEN)


def test_sequence_outputs_depend_on_single_and_pair_state():
    single, pair, mask = _inputs()
    gate = LassoSequenceGate(12, 10, hidden_dim=24, max_candidates=3).eval()
    baseline = gate(single, pair, mask).no_lasso_logit
    zero_single = gate(torch.zeros_like(single), pair, mask).no_lasso_logit
    zero_pair = gate(single, torch.zeros_like(pair), mask).no_lasso_logit
    for changed in (zero_single, zero_pair):
        relative = (baseline - changed).norm() / baseline.norm().clamp_min(1e-8)
        assert relative > 1e-3


def test_adapter_is_trainable_and_masks_padding():
    single, pair, mask = _inputs()
    mask[:, -2:] = False
    adapter = OpenDDEReasoningAdapter(12, 10, 8, 6)
    out_single, out_pair = adapter(single, pair, mask)
    assert not out_single[:, :-2].eq(0).all()
    assert out_single[:, -2:].eq(0).all()
    assert out_pair[:, -2:].eq(0).all() and out_pair[:, :, -2:].eq(0).all()
    (out_single.square().mean() + out_pair.square().mean()).backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())


def test_three_state_decision_uses_ood_and_two_thresholds():
    probability = torch.tensor([0.05, 0.9, 0.5, 0.9])
    ood = torch.tensor([0.1, 0.1, 0.1, 0.95])
    assert decide_sequence(probability, ood, reject_threshold=0.1, accept_threshold=0.8, ood_threshold=0.9) == [
        "NON_LASSO", "LASSO_PLAUSIBLE", "ABSTAIN", "ABSTAIN"
    ]


def test_acceptor_logits_are_strictly_masked_to_asp_glu_and_plug_excludes_nterm():
    single, pair, mask = _inputs()
    residue_type = torch.tensor([[0, 2, 3, 4, 5, 6, 7], [2, 0, 0, 3, 0, 0, 0]])
    residue_index = torch.arange(7)[None].expand(2, -1)
    output = LassoSequenceGate(12, 10, hidden_dim=24)(single, pair, mask, residue_type, residue_index)
    allowed = (residue_type == 2) | (residue_type == 3)
    assert (output.acceptor_logits[~allowed] < -1e20).all()
    assert (output.acceptor_logits[allowed] > -1e20).all()
    assert (output.plug_logits[:, 0] < -1e20).all()
