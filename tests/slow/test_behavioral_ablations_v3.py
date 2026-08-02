from __future__ import annotations

import os
import pytest


def test_release_ablation_manifest_is_required_for_signed_release():
    manifest = os.environ.get("LASSODIFF_V3_ABLATION_MANIFEST")
    if not manifest:
        pytest.skip("release-only: set LASSODIFF_V3_ABLATION_MANIFEST after locked V3 training")
    import json
    payload = json.load(open(manifest, encoding="utf-8"))
    required = {"opendde_pair", "topology_adapter", "structural_roles", "dynamic_geometry"}
    assert required <= set(payload.get("ablations", {}))
    for name in required:
        assert payload["ablations"][name]["effect_size"] >= payload["ablations"][name]["minimum_effect_size"]
