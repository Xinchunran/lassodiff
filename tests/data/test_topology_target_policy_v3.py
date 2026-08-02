from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.data.topology_targets import (
    assess_topology_target,
    record_has_topology_target,
    select_topology_target,
)


def _conformer(name: str, distance: float):
    coords = torch.zeros(4, 7, 3)
    mask = torch.ones(4, 7, dtype=torch.bool)
    mask[:, 6] = False
    coords[0, 0] = torch.tensor([-distance, 0.0, 0.0])
    coords[1, 1] = torch.tensor([1.0, -1.0, 0.0])
    coords[1, 4] = torch.tensor([0.0, 0.0, 0.0])
    coords[1, 5] = torch.tensor([0.5, 0.8660254, 0.0])
    return {"name": name, "coords": coords, "atom_mask": mask}


def _candidate(rank=1):
    return {"rank": rank, "k": 1, "p": 3, "acceptor_index": 1, "prior": 1.0}


def test_relaxed_valid_target_wins_over_open_minimum():
    opened = _conformer("min1", 4.2)
    relaxed = _conformer("relax1", 1.33)
    assert not assess_topology_target(opened, _candidate()).valid
    quality = assess_topology_target(relaxed, _candidate())
    assert quality.valid
    assert quality.has_second_oxygen is False
    selected, valid = select_topology_target([opened, relaxed], _candidate(), deterministic_index=0)
    assert valid is True
    assert selected["name"] == "relax1"


def test_invalid_candidate_is_explicit_and_record_filter_requires_one_valid_target():
    record = {"candidates": [_candidate()], "conformers": [_conformer("min1", 4.2)]}
    selected, valid = select_topology_target(record["conformers"], record["candidates"][0], deterministic_index=0)
    assert selected["name"] == "min1"
    assert valid is False
    assert record_has_topology_target(record) is False


def test_record_filter_accepts_valid_candidate_without_requiring_all_candidates():
    record = {
        "candidates": [_candidate(1), _candidate(2)],
        "conformers": [_conformer("relax1", 1.33), _conformer("min2", 4.2)],
    }
    assert record_has_topology_target(record) is True
