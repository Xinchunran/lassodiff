import torch
from lassodiff.losses_mini_v2 import all_atom_clash_loss,circular_velocity_loss


def test_velocity_loss_penalizes_extra_winding():
    base = circular_velocity_loss(torch.tensor([0.3]), torch.tensor([0.3]), torch.tensor([True]))
    winding = circular_velocity_loss(torch.tensor([0.3 + 2 * torch.pi]), torch.tensor([0.3]), torch.tensor([True]))
    assert float(base) == 0.0
    assert float(winding) > 1.0


def test_clash_loss_excludes_bonded_pairs():
    x=torch.tensor([[[0.,0,0],[1.,0,0]]]); m=torch.ones(1,2,dtype=torch.bool); b=torch.tensor([[[False,True],[True,False]]]); z=torch.zeros_like(b); assert float(all_atom_clash_loss(x,m,[['C','C']],bonded_12=b,bonded_13=z,bonded_14=z))==0.


def test_clash_loss_detects_nonbonded_overlap():
    x=torch.tensor([[[0.,0,0],[.5,0,0]]]); m=torch.ones(1,2,dtype=torch.bool); z=torch.zeros(1,2,2,dtype=torch.bool); assert float(all_atom_clash_loss(x,m,[['C','C']],bonded_12=z,bonded_13=z,bonded_14=z))>.1
