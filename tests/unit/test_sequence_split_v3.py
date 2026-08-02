from __future__ import annotations

import copy

import pytest

from lassodiff.data.sequence_v3 import (
    SequenceExample,
    build_sequence_split,
    validate_sequence_split_contract,
)


def _positive(identifier, sequence, group, acceptor=2, plug=5):
    return SequenceExample(
        identifier, sequence, 1, "positive", "verified_lasso", group,
        acceptor_index=acceptor, plug_index=plug,
    )


def _negative(identifier, sequence, kind, group):
    return SequenceExample(identifier, sequence, 0, kind, "background_non_lasso", group)


def _examples():
    rows = [
        _positive("positive-a", "ACDEFGHIK", "family-a"),
        _negative("mutant-a", "ACNEFGHIK", "hard_mutant", "family-a"),
        # Different annotated family, but one Hamming edit from positive-a.
        _positive("positive-b", "ACDEFGHIL", "family-b"),
    ]
    backgrounds = [
        "LLLLLLLLL", "MMMMMMMMM", "NNNNNNNNN", "PPPPPPPPP", "QQQQQQQQQ",
        "RRRRRRRRR", "SSSSSSSSS", "TTTTTTTTT", "VVVVVVVVV", "WWWWWWWWW",
        "YYYYYYYYY", "GAGAGAGAG",
    ]
    rows.extend(
        _negative(f"background-{index}", sequence, "background", f"background-{index}")
        for index, sequence in enumerate(backgrounds)
    )
    return rows


def test_sequence_split_binds_source_groups_and_hamming_neighbours():
    examples = _examples()
    contract = build_sequence_split(
        examples, seed=19, val_fraction=0.2, test_fraction=0.2,
        max_sequence_distance=0.2, min_stratum_size=1,
    )
    validate_sequence_split_contract(contract, examples)
    owner = {
        example_id: split for split in ("train", "val", "test")
        for example_id in contract["split"][split]
    }
    assert owner["positive-a"] == owner["mutant-a"]
    assert owner["positive-a"] == owner["positive-b"]
    assert contract["cluster_id_by_example"]["positive-a"] == contract["cluster_id_by_example"]["positive-b"]


def test_sequence_split_stratifies_positive_topology_and_negative_kind():
    examples = _examples()
    contract = build_sequence_split(
        examples, seed=23, val_fraction=0.2, test_fraction=0.2,
        max_sequence_distance=0.0, min_stratum_size=1,
    )
    all_strata = set(contract["stratum_by_example"].values())
    assert any(value.startswith("positive:D:ring=3:gap=1-3") for value in all_strata)
    assert "negative:background:ood=0" in all_strata
    assert "negative:hard_mutant:ood=0" in all_strata


def test_sequence_split_contract_fails_closed_on_tampering():
    examples = _examples()
    contract = build_sequence_split(
        examples, seed=29, val_fraction=0.2, test_fraction=0.2,
        max_sequence_distance=0.2, min_stratum_size=1,
    )
    broken = copy.deepcopy(contract)
    broken["stratum_counts"]["train"] = {}
    with pytest.raises(RuntimeError, match="stratum counts"):
        validate_sequence_split_contract(broken, examples)
