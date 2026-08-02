from __future__ import annotations

import pytest
torch=pytest.importorskip("torch")

from scripts.train_structure_v3 import _center_targets


def test_candidate_targets_are_centered_using_only_valid_atoms():
    coords=torch.randn(2,3,5,7,3)+torch.tensor([30.,-11.,8.])
    mask=torch.ones(2,3,5,7,dtype=torch.bool);mask[:,:,4,3:]=False
    centered=_center_targets(coords,mask);weight=mask[...,None]
    mean=(centered*weight).sum((-3,-2))/weight.sum((-3,-2)).clamp(min=1)
    torch.testing.assert_close(mean,torch.zeros_like(mean),atol=1e-5,rtol=0)
    assert centered[~mask].eq(0).all()
