from __future__ import annotations

import json

import pytest

from scripts.verify_v3_startup import verify_startup


def test_startup_contract_accepts_real_manifests():
    result = verify_startup(
        "configs/lassodiff_opendde_v3.yaml",
        "data/lassopred.lmdb/split_topology_hamming_v4.json",
        "artifacts/v3/preflight.real.json",
        "artifacts/v3/cache_manifest.json",
        "artifacts/v3/threading_alignment_v3.json",
    )
    assert result["status"] == "PASS"
    assert result["global_batch_size"] == 64
    assert result["threading_checker_version"] == "v3.2.p_tail_centroid_fan_v1"


def test_startup_contract_rejects_alignment_split_mismatch(tmp_path):
    report = json.loads(open("artifacts/v3/threading_alignment_v3.json", encoding="utf-8").read())
    report["split_manifest_sha256"] = "bad"
    import hashlib
    report["report_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in report.items() if key != "report_sha256"},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="split hash"):
        verify_startup(
            "configs/lassodiff_opendde_v3.yaml",
            "data/lassopred.lmdb/split_topology_hamming_v4.json",
            "artifacts/v3/preflight.real.json",
            "artifacts/v3/cache_manifest.json",
            path,
        )
