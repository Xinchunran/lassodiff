import copy

import pytest

from lassodiff.data.family_split import build_family_split
from lassodiff.data.mini_split import (
    build_mini_cv_manifest, select_mini_cv_fold, validate_mini_cv_manifest,
)


def _source_split():
    records = {}
    topologies = {}
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for index in range(50):
        # Unique equal-length sequences avoid accidental giant single-linkage clusters.
        sequence = "D" + "".join(alphabet[(index * 7 + offset * 3) % len(alphabet)] for offset in range(11))
        record = f"LP_{index:04d}"
        records[record] = sequence
        topologies[record] = {
            "candidates": [{"rank": 1, "k": 4 + index % 4, "p": 6 + index % 4 + index % 3, "acceptor_type": "D"}],
        }
    return build_family_split(
        records, topologies, seed=3, val_fraction=.1, test_fraction=.1,
        max_sequence_distance=.01, min_stratum_size=2,
    )


def test_mini_cv_locks_test_and_holds_each_development_cluster_once():
    source = _source_split()
    unavailable = [source["train"][0], source["test"][0]]
    manifest = build_mini_cv_manifest(source, unavailable_record_ids=unavailable, seed=19)
    validate_mini_cv_manifest(manifest, source)
    assert manifest["locked_test"] == sorted(source["test"])
    assert manifest["unavailable_record_ids"] == sorted(unavailable)
    owners = {}
    for fold_index in range(5):
        fold = select_mini_cv_fold(manifest, fold_index)
        assert not set(fold["train"]) & set(fold["val"])
        assert not set(fold["train"] + fold["val"]) & set(source["test"])
        for record in fold["val"]:
            owners[record] = owners.get(record, 0) + 1
    assert set(owners) == set(source["train"]) | set(source["val"])
    assert set(owners.values()) == {1}


def test_mini_cv_rejects_test_movement_and_hash_tampering():
    source = _source_split()
    manifest = build_mini_cv_manifest(source)
    moved = copy.deepcopy(manifest)
    moved["locked_test"].pop()
    with pytest.raises(RuntimeError, match="locked test"):
        validate_mini_cv_manifest(moved, source)
    corrupted = copy.deepcopy(manifest)
    corrupted["seed"] += 1
    with pytest.raises(RuntimeError, match="manifest hash"):
        validate_mini_cv_manifest(corrupted, source)
