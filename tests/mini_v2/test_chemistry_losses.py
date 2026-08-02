import torch
from lassodiff.losses_mini_v2 import all_atom_clash_loss,circular_velocity_loss


def test_circular_velocity_loss_respects_periodicity():
    assert float(circular_velocity_loss(torch.tensor([torch.pi-.01]),torch.tensor([-torch.pi+.01]),torch.tensor([True])))<1e-3


def test_clash_loss_excludes_bonded_pairs():
    x=torch.tensor([[[0.,0,0],[1.,0,0]]]); m=torch.ones(1,2,dtype=torch.bool); b=torch.tensor([[[False,True],[True,False]]]); z=torch.zeros_like(b); assert float(all_atom_clash_loss(x,m,[['C','C']],bonded_12=b,bonded_13=z,bonded_14=z))==0.


def test_clash_loss_detects_nonbonded_overlap():
    x=torch.tensor([[[0.,0,0],[.5,0,0]]]); m=torch.ones(1,2,dtype=torch.bool); z=torch.zeros(1,2,2,dtype=torch.bool); assert float(all_atom_clash_loss(x,m,[['C','C']],bonded_12=z,bonded_13=z,bonded_14=z))>.1
