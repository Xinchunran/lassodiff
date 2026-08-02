from __future__ import annotations

import copy

import pytest
import torch

from lassodiff.threading_truth import (
    THREADING_CHECKER_VERSION,
    _pairwise_auc,
    unthreaded_decoy,
    validate_threading_alignment_report,
    wrong_direction_decoy,
)
from lassodiff.topology_threading import _hard_candidate, _soft_candidate


def _ring_case():
    ring = torch.tensor([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
    ca = torch.zeros(7, 3)
    ca[:4] = ring
    ca[4:] = torch.tensor([[0., 0., -1.], [0., 0., 1.], [0., 0., 2.]])
    return ca


def test_decoys_have_expected_authoritative_labels():
    ca = _ring_case()
    assert _hard_candidate(ca, 3, 4)[1] == 1
    assert _hard_candidate(unthreaded_decoy(ca, 3, 4), 3, 4)[1] == 0
    reflected = wrong_direction_decoy(ca, 3, 4)
    original = _hard_candidate(ca, 3, 4)
    flipped = _hard_candidate(reflected, 3, 4)
    assert flipped[1] == 1
    assert flipped[2] == -original[2]


def test_soft_score_has_gradient_and_pairwise_auc():
    ca = _ring_case().requires_grad_()
    decoy = unthreaded_decoy(ca.detach(), 3, 4)
    kwargs = {"plane_temperature": .25, "barycentric_temperature": .05, "segment_temperature": .05}
    positive = abs(float(_soft_candidate(ca, 3, 4, **kwargs)))
    negative = abs(float(_soft_candidate(decoy, 3, 4, **kwargs)))
    assert positive > negative
    loss = _soft_candidate(ca, 3, 4, **kwargs)
    loss.backward()
    assert ca.grad is not None and torch.isfinite(ca.grad).all()
    assert _pairwise_auc([positive], [negative]) == pytest.approx(1.0)


def test_alignment_report_validation_fails_closed():
    report = {
        "checker_version": THREADING_CHECKER_VERSION,
        "chemistry_qualified_target_count": 100,
        "threaded_target_rate": .95,
        "soft_pairwise_auc_unthreaded": .95,
        "wrong_direction_checked": 100,
        "wrong_direction_class_flip_rate": .95,
    }
    import hashlib, json
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    validate_threading_alignment_report(report)
    broken = copy.deepcopy(report)
    broken["soft_pairwise_auc_unthreaded"] = .2
    with pytest.raises(RuntimeError, match="AUROC"):
        validate_threading_alignment_report(broken)
