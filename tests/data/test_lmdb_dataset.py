from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lmdb")

from lassodiff.data.lassopred_lmdb import (
    ESMEmbeddingCache,
    LassoPredLMDBDataset,
    build_lassopred_lmdb,
    make_split_manifest,
    write_esm_cache,
)
from tests.data.test_lassopred_dataset import _metadata, _write_pdb


def test_lmdb_roundtrip_and_reject_report(tmp_path: Path):
    structures = tmp_path / "structures"
    entry = structures / "LP_TEST"
    entry.mkdir(parents=True)
    _write_pdb(entry / "min1.pdb")
    _write_pdb(entry / "min2.pdb")
    _write_pdb(entry / "relax2.pdb")
    metadata = tmp_path / "data.json"
    metadata.write_text(json.dumps([_metadata(), {"LP_ID": "BROKEN"}]), encoding="utf-8")
    output = tmp_path / "dataset.lmdb"

    manifest = build_lassopred_lmdb(metadata, structures, output)
    dataset = LassoPredLMDBDataset(output)

    assert manifest["record_count"] == len(dataset) == 1
    item = dataset[0]
    assert item["target_names"][0] == "min1"
    assert item["target_names"][1] in {"min2", "relax2"}
    assert item["coords"].shape == (2, 4, 7, 3)
    assert item["atom_mask"].shape == (2, 4, 7)
    # Every target is paired with the candidate of the same numbered rank.
    assert [int(name[-1]) for name in item["target_names"]] == [1, 2]
    rejects = (tmp_path / "dataset.rejects.jsonl").read_text(encoding="utf-8")
    assert "BROKEN" in rejects


def test_split_has_no_record_overlap():
    split = make_split_manifest(["a", "b", "c", "d", "e"], seed=9, val_fraction=0.2, test_fraction=0.2)
    groups = [set(split[key]) for key in ("train", "val", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == {"a", "b", "c", "d", "e"}


def test_esm_cache_is_sequence_and_model_specific(tmp_path: Path):
    cache_path = tmp_path / "esm.lmdb"
    write_esm_cache(cache_path, "esm2_t33_650M_UR50D", 33, [("ADGY", torch.ones(4, 1280))])
    cache = ESMEmbeddingCache(cache_path, "esm2_t33_650M_UR50D", 33)
    assert cache.get("ADGY").shape == (4, 1280)
    with pytest.raises(KeyError, match="cache miss"):
        cache.get("AAAA")
    with pytest.raises(Exception, match="mismatch"):
        ESMEmbeddingCache(cache_path, "esm2_t12_35M_UR50D", 12)
