from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.sequence_gate import SequenceAssessment
from lassodiff.sequence_objective import sequence_gate_objective


def _assessment():
    return SequenceAssessment(
        no_lasso_logit=torch.tensor([-2., 2.], requires_grad=True),
        acceptor_logits=torch.randn(2, 5, requires_grad=True), plug_logits=torch.randn(2, 5, requires_grad=True),
        candidate_logits=torch.tensor([[2., 0., -1.], [0., 0., 0.]], requires_grad=True),
        sequence_embedding=torch.randn(2, 4, requires_grad=True),
        ood_score=torch.tensor([.1, .9], requires_grad=True),
    )


def test_hypothesis_loss_uses_nonuniform_teacher_prior():
    assessment = _assessment()
    base = dict(label=torch.tensor([1., 0.]), token_mask=torch.ones(2, 5, dtype=torch.bool),
                acceptor_index=torch.tensor([1, -1]), plug_index=torch.tensor([4, -1]),
                candidate_mask=torch.tensor([[True, True, True], [False, False, False]]),
                ood_target=torch.tensor([0., 1.]))
    left = sequence_gate_objective(assessment, teacher_prior=torch.tensor([[.9, .09, .01], [0., 0., 0.]]), **base)
    right = sequence_gate_objective(assessment, teacher_prior=torch.tensor([[.01, .09, .9], [0., 0., 0.]]), **base)
    assert left["hypothesis"] < right["hypothesis"]


def test_negative_does_not_receive_acceptor_plug_or_candidate_loss():
    assessment = _assessment()
    result = sequence_gate_objective(
        assessment, label=torch.tensor([1., 0.]), token_mask=torch.ones(2, 5, dtype=torch.bool),
        acceptor_index=torch.tensor([1, -1]), plug_index=torch.tensor([4, -1]),
        teacher_prior=torch.tensor([[.8, .1, .1], [0., 0., 0.]]),
        candidate_mask=torch.tensor([[True, True, True], [False, False, False]]),
        ood_target=torch.tensor([0., 1.]),
    )
    result["total"].backward()
    assert torch.isfinite(result["total"])
    assert torch.isfinite(assessment.candidate_logits.grad).all()
