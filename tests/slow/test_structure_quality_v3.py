from __future__ import annotations

import json, os
import pytest


def test_signed_structure_release_comparison_meets_contract():
    path=os.environ.get("LASSODIFF_V3_STRUCTURE_RELEASE_METRICS")
    if not path:
        pytest.skip("release-only: set locked V2/V3 rollout comparison after training")
    report=json.load(open(path,encoding="utf-8")); baseline=report["baseline"]; v3=report["v3"]
    assert v3["threading_success"] > baseline["threading_success"]
    assert v3["iso_validity"] >= baseline["iso_validity"]
    assert v3["clash_rate"] <= baseline["clash_rate"]
    assert v3["ca_rmsd_delta_ci95_upper"] <= report["locked_margins"]["ca_rmsd"]
    assert v3["ca_lddt_delta_ci95_lower"] >= -report["locked_margins"]["ca_lddt"]
