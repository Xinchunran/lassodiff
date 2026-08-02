from __future__ import annotations

import json, os
import pytest


def test_signed_sequence_release_metrics_meet_locked_thresholds():
    path = os.environ.get("LASSODIFF_V3_SEQUENCE_RELEASE_METRICS")
    if not path:
        pytest.skip("release-only: verified positive/background sequence dataset is not yet supplied")
    metrics=json.load(open(path,encoding="utf-8")); thresholds=metrics["locked_thresholds"]
    assert metrics["random_acceptance_rate"] <= .01
    assert metrics["composition_shuffle_fpr"] <= .05
    for key in ("hard_mutant_fpr","overall_fpr_at_target_recall","ece","ood_false_acceptance"):
        assert metrics[key] <= thresholds[key]
