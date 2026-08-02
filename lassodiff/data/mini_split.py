"""Locked-test, topology-stratified cluster cross-validation for Mini."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random

from .family_split import validate_family_split_manifest


MINI_CV_SPLIT_VERSION = 1
MINI_CV_SPLIT_METHOD = "locked_v3_test_topology_stratified_cluster_5fold_v1"


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _allocation_cost(counts, target, strata, stratum_targets) -> float:
    cost = sum((count - target) ** 2 / max(target, 1.0) for count in counts)
    for fold_counts in strata:
        for name, expected in stratum_targets.items():
            cost += (fold_counts[name] - expected) ** 2 / max(expected, 1.0)
    return cost


def build_mini_cv_manifest(
    source_split: dict, *, unavailable_record_ids=(), fold_count: int = 5, seed: int = 17,
) -> dict:
    """Pool source train/val into cluster-isolated folds; never alter source test."""
    validate_family_split_manifest(source_split)
    if fold_count != 5:
        raise ValueError("Mini cross-validation contract requires exactly five folds")
    source_ids = set(source_split["train"]) | set(source_split["val"]) | set(source_split["test"])
    unavailable = sorted(set(unavailable_record_ids))
    if not set(unavailable) <= source_ids:
        raise ValueError("unavailable Mini records must belong to the source split")

    development = sorted(set(source_split["train"]) | set(source_split["val"]))
    locked_test = sorted(source_split["test"])
    cluster_by_record = source_split["cluster_id_by_record"]
    stratum_by_record = source_split["topology_stratum_by_record"]
    grouped: dict[str, list[str]] = {}
    for record_id in development:
        grouped.setdefault(cluster_by_record[record_id], []).append(record_id)
    groups = [(cluster, sorted(records)) for cluster, records in grouped.items()]
    total_strata = Counter(stratum_by_record[record] for record in development)
    target_count = len(development) / fold_count
    stratum_targets = {name: count / fold_count for name, count in total_strata.items()}
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda item: (
        -len(item[1]),
        min(total_strata[stratum_by_record[record]] for record in item[1]),
    ))

    folds = [[] for _ in range(fold_count)]
    fold_counts = [0] * fold_count
    fold_strata = [Counter() for _ in range(fold_count)]
    fold_by_cluster = {}
    for cluster, records in groups:
        group_strata = Counter(stratum_by_record[record] for record in records)
        choices = []
        for fold in range(fold_count):
            candidate_counts = list(fold_counts)
            candidate_counts[fold] += len(records)
            candidate_strata = [value.copy() for value in fold_strata]
            candidate_strata[fold].update(group_strata)
            choices.append((
                _allocation_cost(candidate_counts, target_count, candidate_strata, stratum_targets),
                fold,
            ))
        destination = min(choices)[1]
        folds[destination].extend(records)
        fold_counts[destination] += len(records)
        fold_strata[destination].update(group_strata)
        fold_by_cluster[cluster] = destination

    fold_payloads = []
    development_set = set(development)
    for fold, validation in enumerate(folds):
        validation = sorted(validation)
        fold_payloads.append({
            "fold": fold,
            "train": sorted(development_set - set(validation)),
            "val": validation,
            "train_record_count": len(development) - len(validation),
            "val_record_count": len(validation),
            "val_topology_stratum_counts": dict(sorted(fold_strata[fold].items())),
        })

    payload = {
        "version": MINI_CV_SPLIT_VERSION,
        "split_method": MINI_CV_SPLIT_METHOD,
        "fold_count": fold_count,
        "seed": seed,
        "source_split_manifest_sha256": source_split["manifest_sha256"],
        "source_split_method": source_split["split_method"],
        "source_record_count": source_split["record_count"],
        "development_record_count": len(development),
        "locked_test_record_count": len(locked_test),
        "development": development,
        "locked_test": locked_test,
        "unavailable_record_ids": unavailable,
        "cluster_id_by_record": dict(sorted(cluster_by_record.items())),
        "topology_stratum_by_record": dict(sorted(stratum_by_record.items())),
        "fold_by_cluster": dict(sorted(fold_by_cluster.items())),
        "folds": fold_payloads,
    }
    payload["manifest_sha256"] = _canonical_hash(payload)
    return payload


def validate_mini_cv_manifest(manifest: dict, source_split: dict) -> dict:
    """Fail closed on test movement, cluster leakage, fold drift, or hash tampering."""
    validate_family_split_manifest(source_split)
    if manifest.get("version") != MINI_CV_SPLIT_VERSION or manifest.get("split_method") != MINI_CV_SPLIT_METHOD:
        raise RuntimeError("Mini cross-validation split method or schema mismatch")
    if manifest.get("fold_count") != 5 or len(manifest.get("folds", [])) != 5:
        raise RuntimeError("Mini cross-validation requires exactly five folds")
    if manifest.get("source_split_manifest_sha256") != source_split.get("manifest_sha256"):
        raise RuntimeError("Mini CV source split hash mismatch")

    development = sorted(set(source_split["train"]) | set(source_split["val"]))
    locked_test = sorted(source_split["test"])
    if manifest.get("development") != development:
        raise RuntimeError("Mini CV development pool differs from source train+val")
    if manifest.get("locked_test") != locked_test:
        raise RuntimeError("Mini CV locked test differs from the V3 test holdout")
    if set(development) & set(locked_test):
        raise RuntimeError("Mini CV development and test sets overlap")
    if manifest.get("cluster_id_by_record") != source_split.get("cluster_id_by_record"):
        raise RuntimeError("Mini CV cluster assignments differ from the source split")
    if manifest.get("topology_stratum_by_record") != source_split.get("topology_stratum_by_record"):
        raise RuntimeError("Mini CV topology strata differ from the source split")
    unavailable = manifest.get("unavailable_record_ids", [])
    if unavailable != sorted(set(unavailable)) or not set(unavailable) <= set(development) | set(locked_test):
        raise RuntimeError("Mini CV unavailable record list is invalid")

    cluster_by_record = source_split["cluster_id_by_record"]
    stratum_by_record = source_split["topology_stratum_by_record"]
    validation_owner = {}
    development_set = set(development)
    expected_fold_by_cluster = {}
    for expected_fold, fold in enumerate(manifest["folds"]):
        if fold.get("fold") != expected_fold:
            raise RuntimeError("Mini CV fold numbering is not canonical")
        train, validation = set(fold.get("train", [])), set(fold.get("val", []))
        if train & validation or train | validation != development_set:
            raise RuntimeError("Mini CV fold does not partition the development pool")
        if set(locked_test) & (train | validation):
            raise RuntimeError("Mini CV fold leaks locked test records")
        if fold.get("train_record_count") != len(train) or fold.get("val_record_count") != len(validation):
            raise RuntimeError("Mini CV fold record count mismatch")
        observed_strata = Counter(stratum_by_record[record] for record in validation)
        if fold.get("val_topology_stratum_counts") != dict(sorted(observed_strata.items())):
            raise RuntimeError("Mini CV topology stratum count mismatch")
        for record in validation:
            if record in validation_owner:
                raise RuntimeError("Mini CV record is validation in more than one fold")
            validation_owner[record] = expected_fold
            cluster = cluster_by_record[record]
            if cluster in expected_fold_by_cluster and expected_fold_by_cluster[cluster] != expected_fold:
                raise RuntimeError("sequence-neighbour cluster crosses Mini validation folds")
            expected_fold_by_cluster[cluster] = expected_fold
    if set(validation_owner) != development_set:
        raise RuntimeError("Mini CV folds do not cover every development record once")
    if manifest.get("fold_by_cluster") != dict(sorted(expected_fold_by_cluster.items())):
        raise RuntimeError("Mini CV cluster-to-fold mapping mismatch")
    if manifest.get("development_record_count") != len(development) or manifest.get("locked_test_record_count") != len(locked_test):
        raise RuntimeError("Mini CV top-level record count mismatch")
    if manifest.get("source_record_count") != source_split.get("record_count"):
        raise RuntimeError("Mini CV source record count mismatch")
    supplied_hash = manifest.get("manifest_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if supplied_hash != _canonical_hash(unhashed):
        raise RuntimeError("Mini CV manifest hash mismatch")
    return manifest


def select_mini_cv_fold(manifest: dict, fold: int) -> dict:
    if not 0 <= int(fold) < 5:
        raise ValueError("Mini CV fold must be in [0,4]")
    selected = manifest["folds"][int(fold)]
    if selected.get("fold") != int(fold):
        raise RuntimeError("Mini CV fold ordering is corrupted")
    return selected
