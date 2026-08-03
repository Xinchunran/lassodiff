import hashlib
import json

import pytest

from lassodiff.data.mini_grouped_pdb_dataset import GroupedMiniPDBDataset
from lassodiff.target_fit_mini_v2 import (
    DECODER_FIT_VERSION,
    load_decoder_fit_manifest,
)


def _write_manifest(root, *, fit_version=DECODER_FIT_VERSION):
    unsigned = {
        "fit_version": fit_version,
        "targets": [{"cache_key": "a" * 64, "converged": True}],
    }
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps({**unsigned, "manifest_sha256": digest}),
        encoding="utf-8",
    )
    return digest


def test_decoder_fit_manifest_is_content_addressed(tmp_path):
    digest = _write_manifest(tmp_path)
    assert load_decoder_fit_manifest(tmp_path)["manifest_sha256"] == digest

    payload = json.loads((tmp_path / "manifest.json").read_text())
    payload["targets"][0]["converged"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="SHA256"):
        load_decoder_fit_manifest(tmp_path)


def test_production_dataset_requires_valid_fit_manifest(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text("[]", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(RuntimeError, match="manifest"):
        GroupedMiniPDBDataset(
            metadata,
            tmp_path,
            decoder_fit_cache=cache,
            require_decoder_fit=True,
        )


def test_decoder_fit_manifest_rejects_schema_drift(tmp_path):
    _write_manifest(tmp_path, fit_version="future_decoder_fit")
    with pytest.raises(RuntimeError, match="version"):
        load_decoder_fit_manifest(tmp_path)
