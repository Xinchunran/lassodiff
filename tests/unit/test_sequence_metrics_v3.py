from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.sequence_metrics import calibrate_gate_thresholds, sequence_classification_metrics


def test_sequence_metrics_report_abstention_without_counting_it_correct():
    probability = torch.tensor([.95, .8, .55, .2, .05])
    label = torch.tensor([1, 1, 1, 0, 0])
    decisions = ["LASSO_PLAUSIBLE", "LASSO_PLAUSIBLE", "ABSTAIN", "NON_LASSO", "NON_LASSO"]
    metrics = sequence_classification_metrics(probability, label, decisions, bins=5)
    assert metrics["coverage"] == pytest.approx(.8)
    assert metrics["abstain_rate"] == pytest.approx(.2)
    assert metrics["covered_accuracy"] == 1
    assert metrics["auroc"] == 1


def test_accept_threshold_is_calibrated_on_negatives_for_target_fpr():
    probability = torch.tensor([.99, .9, .8, .7, .4, .3, .2, .1])
    label = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    thresholds = calibrate_gate_thresholds(probability, label, target_fpr=.25, target_recall=.75)
    negative_fpr = float((probability[label == 0] >= thresholds["accept_threshold"]).float().mean())
    assert negative_fpr <= .25
    assert thresholds["reject_threshold"] < thresholds["accept_threshold"]
