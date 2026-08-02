import json
import pytest
from scripts.verify_mini_v2_startup import _require_v2_preflight


def test_v2_preflight_rejects_legacy_identity(tmp_path):
    path = tmp_path / "preflight.json"; path.write_text(json.dumps({"status": "PASS", "architecture_id": "lassodiff_mini_core7", "schema_version": 1}))
    with pytest.raises(RuntimeError):
        _require_v2_preflight(path, {"manifest_sha256": "a"}, {"manifest_sha256": "b"})
