from __future__ import annotations

import copy
import pytest

from lassodiff.data.family_split import build_family_split, validate_family_split_manifest


def _topologies(records):
    return {
        record: [{"k": 3, "p": 6, "acceptor_type": "ASP", "rank": 1}]
        for record in records
    }


def test_similar_sequences_and_duplicate_records_never_cross_split():
    records = {
        "a1": "ACDEFGHIKLMN", "a2": "ACDEFGHIKLMN", "a3": "ACDEYGHIKLMN",
        "b1": "PPPPPPPPPPPP", "b2": "PPPPPPPPPPPA",
        "c1": "WYRQTSVNADKC", "d1": "GGGGAAAATTTT",
    }
    manifest = build_family_split(
        records, _topologies(records), seed=3, val_fraction=.2,
        test_fraction=.2, max_sequence_distance=.3,
    )
    owner = {record: split for split in ("train", "val", "test") for record in manifest[split]}
    assert owner["a1"] == owner["a2"] == owner["a3"]
    assert owner["b1"] == owner["b2"]
    assert manifest["cluster_id_by_record"]["a1"] == manifest["cluster_id_by_record"]["a3"]
    assert sorted(owner) == sorted(records)
    assert manifest["split_method"] == "topology_stratified_hamming_lcs_single_linkage_v2"


def test_family_split_is_deterministic_and_has_manifest_hash():
    records = {f"id{i}": sequence for i, sequence in enumerate((
        "ACDEFGHIK", "ACDEYGHIK", "PPPPPPPPA", "PPPPPPPPG", "WQRTYIPAS", "GAVLNKMST",
    ))}
    first = build_family_split(records, _topologies(records), seed=17, val_fraction=.2, test_fraction=.2)
    second = build_family_split(records, _topologies(records), seed=17, val_fraction=.2, test_fraction=.2)
    assert first == second
    assert len(first["manifest_sha256"]) == 64


def test_family_manifest_validator_rejects_cluster_leakage():
    records={"a":"ACDEFGHIK","b":"ACDEYGHIK","c":"PPPPPPPPA","d":"WQRTYIPAS"}
    manifest=build_family_split(
        records, _topologies(records), seed=2, val_fraction=.25,
        test_fraction=.25, max_sequence_distance=.3,
    )
    validate_family_split_manifest(manifest)
    broken=copy.deepcopy(manifest)
    cluster=broken["cluster_id_by_record"]["a"]
    mate=next(record for record,value in broken["cluster_id_by_record"].items() if value==cluster and record!="a")
    source=next(name for name in ("train","val","test") if mate in broken[name])
    destination=next(name for name in ("train","val","test") if name!=source)
    broken[source].remove(mate);broken[destination].append(mate)
    with pytest.raises(RuntimeError,match="cluster crosses|cluster crosses split"):
        validate_family_split_manifest(broken)
