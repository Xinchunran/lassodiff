import torch
from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.dynamic_geometry_mini import compute_dynamic_pair_geometry
from lassodiff.model_mini_v2 import MiniTorsionDiffusion
from lassodiff.torsion_state import TorsionState


def test_dynamic_geometry_is_global_rigid_transform_invariant():
    x=torch.randn(1,8,7,3); m=torch.ones(1,8,dtype=torch.bool); a=compute_dynamic_pair_geometry(x,m); r=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]); b=compute_dynamic_pair_geometry(x@r.T+torch.tensor([4.,-3.,2.]),m); assert torch.allclose(a,b,atol=2e-5,rtol=2e-5)


def test_model_recomputes_geometry_once_per_block(fake_conditioning):
    c=CandidateCondition('AAADRAAA',3,5); s=TorsionState(torch.zeros(1,1,8,3),torch.ones(1,1,8,3,dtype=torch.bool),torch.zeros(1,1,8,4),torch.zeros(1,1,8,4,dtype=torch.bool)); out=MiniTorsionDiffusion(single_dim=32,pair_dim=16,hidden_dim=32,blocks=3)(state_t=s,time=torch.tensor([.5]),conditioning=fake_conditioning,token_mask=torch.ones(1,8,dtype=torch.bool),candidates=[c]); assert out.dynamic_geometry_calls==3
