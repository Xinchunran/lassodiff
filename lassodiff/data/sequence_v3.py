"""Versioned sequence-gate examples with evidence and leakage contracts."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .family_split import (
    _UnionFind, _allocation_cost, _candidate_pairs, normalized_sequence_distance,
)


@dataclass(frozen=True)
class SequenceExample:
    example_id: str
    sequence: str
    label: int
    kind: str
    evidence: str
    group_id: str
    acceptor_index: int | None = None
    plug_index: int | None = None
    ood: bool = False
    teacher_prior: tuple[float, ...] | list[float] | None = None


def validate_sequence_examples(examples, source_groups=None):
    if not examples:
        raise RuntimeError("sequence gate dataset is empty")
    identifiers = set()
    for row in examples:
        if row.example_id in identifiers:
            raise RuntimeError("duplicate sequence example id")
        identifiers.add(row.example_id)
        if row.label not in (0, 1) or not row.sequence or not row.group_id:
            raise RuntimeError("malformed sequence example")
        if any(residue not in "ACDEFGHIKLMNPQRSTVWY" for residue in row.sequence):
            raise RuntimeError("sequence examples must contain canonical amino acids")
        if row.label == 1 and row.evidence not in {"verified_lasso", "experimental_lasso"}:
            raise RuntimeError("sequence positive lacks verified positive evidence")
        if row.label == 1:
            if row.acceptor_index is None or not 0 <= row.acceptor_index < len(row.sequence):
                raise RuntimeError("positive sequence is missing a legal acceptor index")
            if row.sequence[row.acceptor_index] not in "DE":
                raise RuntimeError("positive sequence acceptor must be ASP/GLU")
            if row.plug_index is None or not 0 < row.plug_index < len(row.sequence):
                raise RuntimeError("positive sequence is missing a legal non-terminal plug index")
        if row.label == 1 and row.teacher_prior is not None:
            if not row.teacher_prior or any(value <= 0 for value in row.teacher_prior):
                raise RuntimeError("positive teacher prior must contain positive probabilities")
        if row.kind == "hard_mutant" and source_groups is not None:
            expected = source_groups.get(row.example_id)
            if expected is None or expected != row.group_id:
                raise RuntimeError("hard mutant does not share its source group")
    return examples


def _gap_bin(gap: int) -> str:
    if gap <= 3:
        return "1-3"
    if gap <= 6:
        return "4-6"
    if gap <= 10:
        return "7-10"
    return "11+"


def _sequence_stratum(row: SequenceExample) -> str:
    if row.label == 0:
        return f"negative:{row.kind}:ood={int(row.ood)}"
    acceptor_type = row.sequence[int(row.acceptor_index)]
    ring_length = int(row.acceptor_index) + 1
    plug_gap = int(row.plug_index) - int(row.acceptor_index)
    return f"positive:{acceptor_type}:ring={ring_length}:gap={_gap_bin(plug_gap)}"


def build_sequence_split(
    examples, *, seed: int = 1701, val_fraction: float = 0.1,
    test_fraction: float = 0.1, max_sequence_distance: float = 0.4,
    min_stratum_size: int = 5,
):
    """Keep source/family and Hamming/LCS neighbours in one split."""
    validate_sequence_examples(examples)
    if min(val_fraction, test_fraction) < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("invalid sequence split fractions")
    if not 0 <= max_sequence_distance < 1 or min_stratum_size < 1:
        raise ValueError("invalid sequence split distance/stratum threshold")
    sequences = sorted({row.sequence for row in examples})
    index = {sequence: position for position, sequence in enumerate(sequences)}
    union = _UnionFind(len(sequences))
    compared = 0
    for left, right in _candidate_pairs(sequences, max_sequence_distance):
        compared += 1
        if normalized_sequence_distance(sequences[left], sequences[right]) <= max_sequence_distance:
            union.union(left, right)
    # Family/source groups are an additional hard edge.  In particular, a
    # positive and every derived shuffle/mutant remain together even when the
    # mutation or shuffle falls outside the distance threshold.
    by_source = {}
    for row in examples:
        by_source.setdefault(row.group_id, []).append(index[row.sequence])
    for members in by_source.values():
        for other in members[1:]:
            union.union(members[0], other)
    grouped = {}
    for row in examples:
        grouped.setdefault(union.find(index[row.sequence]), []).append(row.example_id)
    cluster_id_by_example = {}
    groups = []
    for number, (_root, members) in enumerate(sorted(grouped.items()), start=1):
        members = sorted(members)
        groups.append(members)
        for example_id in members:
            cluster_id_by_example[example_id] = f"sequence-cluster-{number:06d}"
    by_id = {row.example_id: row for row in examples}
    strata = {row.example_id: _sequence_stratum(row) for row in examples}
    total_strata = Counter(strata.values())
    eligible = {name for name, count in total_strata.items() if count >= min_stratum_size}
    fractions = {"train": 1.0 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    targets = {name: len(examples) * fraction for name, fraction in fractions.items()}
    stratum_targets = {
        name: {key: count * fraction for key, count in total_strata.items() if key in eligible}
        for name, fraction in fractions.items()
    }
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda members: (-len(members), min(total_strata[strata[item]] for item in members)))
    split = {"train": [], "val": [], "test": []}
    counts = Counter({name: 0 for name in split})
    split_strata = {name: Counter() for name in split}
    order = ("train", "val", "test")
    for members in groups:
        group_strata = Counter(strata[item] for item in members)
        choices = []
        for destination in order:
            candidate_counts = counts.copy()
            candidate_counts[destination] += len(members)
            candidate_strata = {name: value.copy() for name, value in split_strata.items()}
            candidate_strata[destination].update(group_strata)
            choices.append((
                _allocation_cost(candidate_counts, targets, candidate_strata, stratum_targets),
                order.index(destination), destination,
            ))
        destination = min(choices)[2]
        split[destination].extend(members)
        counts[destination] += len(members)
        split_strata[destination].update(group_strata)
    contract = {
        "version": 2,
        "method": "label_topology_stratified_hamming_lcs_group_split_v2",
        "sequence_distance": "Hamming/max_length for equal length; 1-LCS/max_length for indels",
        "max_sequence_distance": max_sequence_distance,
        "min_stratum_size": min_stratum_size,
        "seed": seed, "val_fraction": val_fraction, "test_fraction": test_fraction,
        "example_count": len(examples), "unique_sequence_count": len(sequences),
        "cluster_count": len(groups), "compared_pair_count": compared,
        "cluster_id_by_example": dict(sorted(cluster_id_by_example.items())),
        "stratum_by_example": dict(sorted(strata.items())),
        "stratum_counts": {name: dict(sorted(split_strata[name].items())) for name in order},
        "split": {name: sorted(split[name]) for name in order},
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    contract["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return contract


def validate_sequence_split_contract(contract, examples):
    if contract.get("version") != 2 or contract.get("method") != "label_topology_stratified_hamming_lcs_group_split_v2":
        raise RuntimeError("sequence topology/Hamming split schema mismatch")
    by_id = {row.example_id: row for row in examples}
    split = contract.get("split", {})
    owner = {}
    for name in ("train", "val", "test"):
        for example_id in split.get(name, []):
            if example_id in owner:
                raise RuntimeError("sequence example crosses split")
            owner[example_id] = name
    if set(owner) != set(by_id):
        raise RuntimeError("sequence split does not cover every example")
    clusters = contract.get("cluster_id_by_example", {})
    strata = contract.get("stratum_by_example", {})
    if set(clusters) != set(by_id) or set(strata) != set(by_id):
        raise RuntimeError("sequence split lacks neighbour/topology assignments")
    cluster_owner = {}
    observed = {name: Counter() for name in ("train", "val", "test")}
    for example_id in by_id:
        name, cluster = owner[example_id], clusters[example_id]
        if cluster in cluster_owner and cluster_owner[cluster] != name:
            raise RuntimeError("sequence neighbour/source cluster crosses split")
        cluster_owner[cluster] = name
        observed[name][strata[example_id]] += 1
    if len(cluster_owner) != contract.get("cluster_count"):
        raise RuntimeError("sequence split cluster count mismatch")
    for name in observed:
        if dict(sorted(observed[name].items())) != contract.get("stratum_counts", {}).get(name):
            raise RuntimeError("sequence split stratum counts mismatch")
    unhashed = {key: value for key, value in contract.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != contract.get("manifest_sha256"):
        raise RuntimeError("sequence split manifest hash mismatch")
    return contract


def load_sequence_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or not isinstance(payload.get("examples"), list):
        raise RuntimeError("sequence manifest schema mismatch")
    rows = [SequenceExample(**row) for row in payload["examples"]]
    source_groups = payload.get("source_groups", {})
    validate_sequence_examples(rows, source_groups=source_groups)
    contract = payload.get("split_contract")
    if not isinstance(contract, dict) or payload.get("split") != contract.get("split"):
        raise RuntimeError("sequence manifest is missing topology/Hamming split contract")
    validate_sequence_split_contract(contract, rows)
    return payload, rows
