from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lassodiff.data.lassopred_lmdb import build_record, collate_lassopred_v2


def _atom(serial: int, atom: str, res: str, idx: int, x: float) -> str:
    return f"ATOM  {serial:5d} {atom:<4s} {res:>3s} A{idx:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C  \n"


def _write_pdb(path: Path) -> None:
    lines = []
    serial = 1
    for idx, residue in enumerate(("ALA", "ASP", "GLY", "TYR"), start=1):
        for atom in ("N", "CA", "C", "O"):
            lines.append(_atom(serial, atom, residue, idx, serial * 0.1))
            serial += 1
        if residue == "ASP":
            for atom in ("CG", "OD1", "OD2"):
                lines.append(_atom(serial, atom, residue, idx, serial * 0.1))
                serial += 1
    path.write_text("".join(lines) + "END\n", encoding="utf-8")


def _metadata():
    return {
        "LP_ID": "LP_TEST", "Core_Sequence": "ADGY", "Core_Length": 4,
        "Ring_Length": 2, "Isopeptide": 2,
        "Upper_Plug_1": 3, "Upper_Plug_2": 4,
    }


def test_all_available_conformers_are_labels_not_min1_relax1_only(tmp_path: Path):
    entry = tmp_path / "LP_TEST"
    entry.mkdir()
    _write_pdb(entry / "min2.pdb")
    _write_pdb(entry / "relax3.pdb")

    record = build_record(_metadata(), tmp_path)

    assert record["sequence"] == "ADGY"
    assert [x["name"] for x in record["conformers"]] == ["min2", "relax3"]
    assert record["iso_acceptor_index"] == 1
    assert record["iso_acceptor_type"] == "ASP"


def test_collate_keeps_variable_candidate_counts_and_masks(tmp_path: Path):
    entry = tmp_path / "LP_TEST"
    entry.mkdir()
    _write_pdb(entry / "min2.pdb")
    first = build_record(_metadata(), tmp_path)
    second = {**first, "record_id": "LP_TEST_2", "candidates": first["candidates"][:1]}

    def sample(record):
        target = record["conformers"][0]
        targets = [target for _ in record["candidates"]]
        return {
            "record_id": record["record_id"], "sequence": record["sequence"],
            "aa_ids": torch.tensor([0, 3, 7, 19]),
            "coords": torch.stack([item["coords"] for item in targets]),
            "atom_mask": torch.stack([item["atom_mask"] for item in targets]),
            "target_names": [item["name"] for item in targets],
            "candidates": record["candidates"],
        }

    batch = collate_lassopred_v2([sample(first), sample(second)])
    assert batch["candidate_mask"].tolist() == [[True, True], [True, False]]
    assert batch["coords"].shape == (2, 2, 4, 7, 3)
    assert batch["atom_mask"].shape == (2, 2, 4, 7)
    assert not batch["atom_mask"][1, 1].any()
    assert batch["target_names"] == [["min2", "min2"], ["min2", None]]
    assert torch.allclose(batch["candidate_prior"].sum(dim=1), torch.ones(2))


def test_sequence_conflict_is_rejected(tmp_path: Path):
    entry = tmp_path / "LP_TEST"
    entry.mkdir()
    _write_pdb(entry / "min1.pdb")
    metadata = _metadata()
    metadata["Core_Sequence"] = "AAAA"
    with pytest.raises(ValueError, match="conflicts"):
        build_record(metadata, tmp_path)


def test_nonfinite_conformer_is_reported_and_not_sampled(tmp_path: Path):
    entry = tmp_path / "LP_TEST"
    entry.mkdir()
    _write_pdb(entry / "min1.pdb")
    bad = (entry / "min1.pdb").read_text(encoding="utf-8").replace("   0.100", "     nan", 1)
    (entry / "min1.pdb").write_text(bad, encoding="utf-8")
    _write_pdb(entry / "min2.pdb")
    record = build_record(_metadata(), tmp_path)
    assert [item["name"] for item in record["conformers"]] == ["min2"]
    assert record["conformer_rejects"] == [{"conformer": "min1.pdb", "reason": "contains non-finite coordinates"}]
