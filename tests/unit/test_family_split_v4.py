from __future__ import annotations

import copy

import pytest

from lassodiff.data.family_split import (
    build_family_split,
    normalized_sequence_distance,
    validate_family_split_manifest,
)


def _topology(k: int, p: int, acceptor: str):
    return [{"k": k, "p": p, "acceptor_type": acceptor, "rank": 1}]


def test_equal_length_distance_is_hamming_and_indels_are_explicit():
    assert normalized_sequence_distance("AAAAAA", "AAAAAT") == pytest.approx(1 / 6)
    assert normalized_sequence_distance("AAAAAA", "AAAAA") == pytest.approx(1 / 6)


def test_hamming_neighbours_and_exact_sequence_records_never_cross_split():
    sequences = {
        "near-a": "AAAAAA", "near-b": "AAAAAT", "same-a": "CCCCCC", "same-b": "CCCCCC",
        "far-a": "GGGGGG", "far-b": "TTTTTT", "far-c": "ACACAC", "far-d": "GTGTGT",
    }
    topologies = {record: _topology(3, 5, "ASP") for record in sequences}
    manifest = build_family_split(
        sequences, topologies, seed=17, val_fraction=0.25, test_fraction=0.25,
        max_sequence_distance=0.2,
    )
    validate_family_split_manifest(manifest)
    owner = {record: split for split in ("train", "val", "test") for record in manifest[split]}
    assert owner["near-a"] == owner["near-b"]
    assert owner["same-a"] == owner["same-b"]
    assert manifest["cluster_id_by_record"]["near-a"] == manifest["cluster_id_by_record"]["near-b"]


def test_topology_strata_are_balanced_without_breaking_clusters():
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    sequences = {}
    topologies = {}
    for index in range(30):
        record = f"asp-{index:02d}"
        sequences[record] = alphabet[index % 20] + alphabet[(index * 7 + 3) % 20] + f"A{index:02d}A"
        topologies[record] = _topology(3, 6, "ASP")
    for index in range(30):
        record = f"glu-{index:02d}"
        sequences[record] = alphabet[(index * 3 + 1) % 20] + alphabet[(index * 11 + 2) % 20] + f"G{index:02d}G"
        topologies[record] = _topology(5, 9, "GLU")
    manifest = build_family_split(
        sequences, topologies, seed=1701, val_fraction=0.2, test_fraction=0.2,
        max_sequence_distance=0.0,
    )
    validate_family_split_manifest(manifest)
    for split, expected in (("train", 18), ("val", 6), ("test", 6)):
        counts = sorted(manifest["topology_stratum_counts"][split].values())
        assert counts == [expected, expected]


def test_manifest_validation_fails_closed_on_cluster_or_stratum_tampering():
    sequences = {f"r{i}": f"AAAA{i}" for i in range(6)}
    topologies = {record: _topology(3, 5, "ASP") for record in sequences}
    manifest = build_family_split(
        sequences, topologies, seed=3, val_fraction=0.2, test_fraction=0.2,
        max_sequence_distance=0.0,
    )
    broken = copy.deepcopy(manifest)
    broken["topology_stratum_counts"]["train"] = {}
    with pytest.raises(RuntimeError, match="stratum counts"):
        validate_family_split_manifest(broken)
