"""Topology-stratified, sequence-neighbour-isolated V3 data split."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
from typing import Mapping, Sequence


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _lcs_length(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for residue in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if residue == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def normalized_sequence_distance(left: str, right: str) -> float:
    """Hamming fraction for equal lengths, indel-aware LCS distance otherwise."""
    left, right = str(left).upper(), str(right).upper()
    if not left or not right:
        raise ValueError("sequence distance requires non-empty sequences")
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) / len(left)
    return 1.0 - _lcs_length(left, right) / max(len(left), len(right))


def _candidate_pairs(sequences: Sequence[str], max_distance: float):
    # Hamming neighbours of equal length must not depend on a shared-kmer
    # heuristic: a short sequence may be close without sharing a 4-mer.
    by_length: dict[int, list[int]] = {}
    for index, sequence in enumerate(sequences):
        by_length.setdefault(len(sequence), []).append(index)
    pairs = set()
    for members in by_length.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1:]:
                pairs.add((left, right))
    # Unequal-length indel neighbours are screened by shared 4-mers before the
    # exact LCS distance.  At the current peptide lengths this is conservative.
    inverted: dict[str, list[int]] = {}
    for index, sequence in enumerate(sequences):
        kmers = {sequence[offset:offset + 4] for offset in range(max(1, len(sequence) - 3))}
        for kmer in kmers:
            inverted.setdefault(kmer, []).append(index)
    for members in inverted.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1:]:
                if len(sequences[left]) == len(sequences[right]):
                    continue
                length_bound = 1.0 - min(len(sequences[left]), len(sequences[right])) / max(
                    len(sequences[left]), len(sequences[right])
                )
                if length_bound <= max_distance:
                    pairs.add((min(left, right), max(left, right)))
    return sorted(pairs)


def _candidate_signatures(value):
    if isinstance(value, Mapping):
        candidates = value.get("candidates", value)
    else:
        candidates = value
    if isinstance(candidates, Mapping):
        candidates = [candidates]
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("topology metadata must provide one or more candidates")
    signatures = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            k, p = int(candidate["k"]), int(candidate["p"])
            acceptor = str(candidate.get("acceptor_type", "UNKNOWN")).upper()
            rank = int(candidate.get("rank", len(signatures) + 1))
        else:
            k, p, acceptor, *rest = candidate
            k, p, acceptor = int(k), int(p), str(acceptor).upper()
            rank = int(rest[0]) if rest else len(signatures) + 1
        signatures.append((rank, k + 1, p - k, acceptor))
    if not signatures:
        raise ValueError("topology metadata has no candidates")
    return sorted(signatures)


def topology_signature(value) -> str:
    """Lossless candidate topology signature retained for audit."""
    return json.dumps(_candidate_signatures(value), separators=(",", ":"))


def topology_stratum(value) -> str:
    """Coarse, statistically supportable topology stratum for allocation."""
    signatures = _candidate_signatures(value)
    ring_lengths = {item[1] for item in signatures}
    acceptors = {item[3] for item in signatures}
    if len(ring_lengths) != 1 or len(acceptors) != 1:
        raise ValueError("candidates in one record must share ring length and acceptor type")
    gaps = [item[2] for item in signatures]

    def gap_bin(gap: int) -> str:
        if gap <= 3:
            return "1-3"
        if gap <= 6:
            return "4-6"
        if gap <= 10:
            return "7-10"
        return "11+"

    return json.dumps(
        (next(iter(acceptors)), next(iter(ring_lengths)), len(signatures), gap_bin(min(gaps)), gap_bin(max(gaps))),
        separators=(",", ":"),
    )


def _allocation_cost(counts, targets, stratum_counts, stratum_targets) -> float:
    cost = 0.0
    for split in ("train", "val", "test"):
        cost += ((counts[split] - targets[split]) ** 2) / max(targets[split], 1.0)
        for stratum, target in stratum_targets[split].items():
            cost += ((stratum_counts[split][stratum] - target) ** 2) / max(target, 1.0)
    return cost


def build_family_split(
    record_sequences: Mapping[str, str], record_topologies: Mapping[str, object], *,
    seed: int = 0, val_fraction: float = 0.1, test_fraction: float = 0.1,
    max_sequence_distance: float = 0.4, min_stratum_size: int = 10,
):
    """Split whole sequence clusters while balancing exact topology strata."""
    if not record_sequences:
        raise ValueError("cannot split an empty sequence dataset")
    if set(record_topologies) != set(record_sequences):
        raise ValueError("every record requires topology metadata for stratification")
    if not 0 <= max_sequence_distance < 1:
        raise ValueError("max_sequence_distance must be in [0,1)")
    if min(val_fraction, test_fraction) < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("invalid split fractions")
    if min_stratum_size < 1:
        raise ValueError("min_stratum_size must be positive")
    sequences = sorted(set(record_sequences.values()))
    sequence_index = {sequence: index for index, sequence in enumerate(sequences)}
    union = _UnionFind(len(sequences))
    compared = 0
    for left, right in _candidate_pairs(sequences, max_sequence_distance):
        compared += 1
        if normalized_sequence_distance(sequences[left], sequences[right]) <= max_sequence_distance:
            union.union(left, right)
    record_clusters: dict[int, list[str]] = {}
    for record_id, sequence in record_sequences.items():
        record_clusters.setdefault(union.find(sequence_index[sequence]), []).append(record_id)
    cluster_id_by_record: dict[str, str] = {}
    groups = []
    for number, (_root, records) in enumerate(sorted(record_clusters.items()), start=1):
        records = sorted(records)
        groups.append(records)
        for record_id in records:
            cluster_id_by_record[record_id] = f"cluster-{number:06d}"

    record_signatures = {record_id: topology_signature(record_topologies[record_id]) for record_id in record_sequences}
    record_strata = {record_id: topology_stratum(record_topologies[record_id]) for record_id in record_sequences}
    total_strata = Counter(record_strata.values())
    eligible_strata = {stratum for stratum, count in total_strata.items() if count >= min_stratum_size}
    fractions = {"train": 1.0 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    targets = {name: len(record_sequences) * fraction for name, fraction in fractions.items()}
    stratum_targets = {
        name: {stratum: count * fraction for stratum, count in total_strata.items() if stratum in eligible_strata}
        for name, fraction in fractions.items()
    }
    rng = random.Random(seed)
    rng.shuffle(groups)
    # Large and topology-rare clusters are placed first; seeded shuffle is the
    # deterministic tie-breaker, not the allocation method.
    groups.sort(key=lambda records: (-len(records), min(total_strata[record_strata[r]] for r in records)))
    split = {"train": [], "val": [], "test": []}
    counts = Counter({name: 0 for name in split})
    strata_by_split = {name: Counter() for name in split}
    destination_order = ("train", "val", "test")
    for group in groups:
        group_strata = Counter(record_strata[record_id] for record_id in group)
        choices = []
        for destination in destination_order:
            candidate_counts = counts.copy()
            candidate_counts[destination] += len(group)
            candidate_strata = {name: value.copy() for name, value in strata_by_split.items()}
            candidate_strata[destination].update(group_strata)
            choices.append((
                _allocation_cost(candidate_counts, targets, candidate_strata, stratum_targets),
                destination_order.index(destination), destination,
            ))
        destination = min(choices)[2]
        split[destination].extend(group)
        counts[destination] += len(group)
        strata_by_split[destination].update(group_strata)

    payload = {
        "version": 4,
        "split_method": "topology_stratified_hamming_lcs_single_linkage_v2",
        "sequence_distance": "Hamming/max_length for equal length; 1-LCS/max_length for indels",
        "max_sequence_distance": max_sequence_distance,
        "candidate_filter": "all_equal_length_pairs; shared_4mer_for_indels",
        "topology_signature_definition": "sorted(rank,ring_length,plug_gap,acceptor_type)",
        "topology_stratum_definition": "acceptor,ring_length,candidate_count,min/max_plug_gap_bin",
        "min_stratum_size": min_stratum_size,
        "seed": seed, "val_fraction": val_fraction, "test_fraction": test_fraction,
        "record_count": len(record_sequences), "unique_sequence_count": len(sequences),
        "cluster_count": len(groups), "compared_pair_count": compared,
        "cluster_id_by_record": dict(sorted(cluster_id_by_record.items())),
        "topology_signature_by_record": dict(sorted(record_signatures.items())),
        "topology_stratum_by_record": dict(sorted(record_strata.items())),
        "topology_stratum_counts": {
            name: dict(sorted(strata_by_split[name].items())) for name in destination_order
        },
        "train": sorted(split["train"]), "val": sorted(split["val"]), "test": sorted(split["test"]),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_family_split_manifest(manifest):
    if manifest.get("version") != 4 or manifest.get("split_method") != "topology_stratified_hamming_lcs_single_linkage_v2":
        raise RuntimeError("V3 topology/Hamming split method or schema mismatch")
    owner = {}
    for split in ("train", "val", "test"):
        for record_id in manifest.get(split, []):
            if record_id in owner:
                raise RuntimeError("record crosses V3 family split")
            owner[record_id] = split
    if len(owner) != manifest.get("record_count"):
        raise RuntimeError("V3 split record count mismatch")
    clusters = manifest.get("cluster_id_by_record", {})
    strata = manifest.get("topology_stratum_by_record", {})
    signatures = manifest.get("topology_signature_by_record", {})
    if set(clusters) != set(owner) or set(strata) != set(owner) or set(signatures) != set(owner):
        raise RuntimeError("V3 split lacks cluster or topology assignments")
    cluster_owner = {}
    observed_strata = {name: Counter() for name in ("train", "val", "test")}
    for record_id, cluster in clusters.items():
        split = owner[record_id]
        if cluster in cluster_owner and cluster_owner[cluster] != split:
            raise RuntimeError("sequence-neighbour cluster crosses split")
        cluster_owner[cluster] = split
        observed_strata[split][strata[record_id]] += 1
    if len(cluster_owner) != manifest.get("cluster_count"):
        raise RuntimeError("V3 split cluster count mismatch")
    expected_strata = manifest.get("topology_stratum_counts", {})
    for split in observed_strata:
        if dict(sorted(observed_strata[split].items())) != expected_strata.get(split):
            raise RuntimeError("V3 split topology stratum counts mismatch")
    supplied_hash = manifest.get("manifest_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if supplied_hash != actual_hash:
        raise RuntimeError("V3 split manifest hash mismatch")
    return manifest
