import torch
from lassodiff.data.mini_grouped_dataset import collate_grouped_mini, group_candidate_examples


def _target(value):
    return {"core": torch.full((8, 7, 3), value), "core_mask": torch.ones((8, 7), dtype=torch.bool), "atom14": torch.full((8, 14, 3), value), "atom14_mask": torch.ones((8, 14), dtype=torch.bool), "backbone_torsions": torch.full((8, 3), value), "backbone_torsion_mask": torch.ones((8, 3), dtype=torch.bool), "chi": torch.full((8, 4), value), "chi_mask": torch.ones((8, 4), dtype=torch.bool)}


def test_same_candidate_ranks_are_grouped():
    rows = [{"record_id":"LP001","sequence":"AAADRAAA","k":3,"p":5,"rank":1,**_target(1.)},{"record_id":"LP001","sequence":"AAADRAAA","k":3,"p":5,"rank":2,**_target(2.)}]
    grouped = group_candidate_examples(rows, max_conformers=3)
    assert len(grouped) == 1 and grouped[0]["core_targets"].shape == (3,8,7,3)
    assert grouped[0]["conformer_mask"].tolist() == [True, True, False]
    assert torch.all(grouped[0]["core_targets"][0] == 1.) and torch.all(grouped[0]["core_targets"][1] == 2.)


def test_different_plugs_are_not_grouped():
    base = {"record_id":"LP001","sequence":"AAADRAAA","k":3,"rank":1}
    assert len(group_candidate_examples([{**base,"p":5,**_target(1.)},{**base,"p":6,**_target(2.)}], 3)) == 2


def test_grouped_collate_preserves_conformer_dimension():
    grouped = group_candidate_examples([{**{"record_id":"LP001","sequence":"AAADRAAA","k":3,"p":5,"rank":1},**_target(1.)},{**{"record_id":"LP001","sequence":"AAADRAAA","k":3,"p":5,"rank":2},**_target(2.)}], 3)
    batch = collate_grouped_mini(grouped)
    assert batch["core_targets"].shape == (1,3,8,7,3) and batch["atom14_targets"].shape == (1,3,8,14,3)
