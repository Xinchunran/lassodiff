import torch
from lassodiff.batch_mini_v2 import prepare_backbone_flow_batch
from lassodiff.data.mini_grouped_dataset import collate_grouped_mini, group_candidate_examples


def _row(value):
    return {"record_id": "x", "sequence": "AAADRAAA", "k": 3, "p": 5, "rank": 1,
            "core": torch.full((8, 7, 3), value), "core_mask": torch.ones(8, 7, dtype=torch.bool),
            "atom14": torch.full((8, 14, 3), value), "atom14_mask": torch.ones(8, 14, dtype=torch.bool),
            "backbone_torsions": torch.full((8, 3), value), "backbone_torsion_mask": torch.ones(8, 3, dtype=torch.bool),
            "chi": torch.full((8, 4), value), "chi_mask": torch.ones(8, 4, dtype=torch.bool)}


def test_flow_batch_uses_grouped_target_and_open_prior():
    batch = collate_grouped_mini(group_candidate_examples([_row(1.0), dict(_row(2.0), rank=2)], 3))
    prepared = prepare_backbone_flow_batch(batch, device=torch.device("cpu"), generator=torch.Generator().manual_seed(7), num_samples=2)
    assert prepared.state_t.backbone.shape[1] == 2
    assert prepared.target_velocity.backbone.abs().sum() > 0
    assert torch.isfinite(prepared.state_t.backbone).all()
