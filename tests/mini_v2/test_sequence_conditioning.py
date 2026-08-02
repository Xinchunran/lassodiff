import torch
import torch.nn as nn
from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.conditioning_mini_v2 import MiniSequenceConditioner
from lassodiff.model_mini_v2 import MiniTorsionDiffusion
from lassodiff.torsion_state import TorsionState


class FakeFrozenEncoder(nn.Module):
    output_dim=8
    def __init__(self): super().__init__(); self.scale=nn.Parameter(torch.ones(())); self.calls=0
    def forward(self,sequences,token_mask):
        self.calls+=1; b,l=token_mask.shape; pos=torch.arange(l,dtype=torch.float32,device=token_mask.device); return pos[None,:,None].expand(b,l,8)*self.scale.detach()*token_mask[...,None]


def test_residue_encoder_is_called_and_frozen():
    e=FakeFrozenEncoder(); c=MiniSequenceConditioner(e,8,32,16); c(sequences=['AAADRAAA'],aa_ids=torch.zeros(1,8,dtype=torch.long),token_mask=torch.ones(1,8,dtype=torch.bool),k=torch.tensor([3]),p=torch.tensor([5])); assert e.calls==1 and not e.scale.requires_grad


def test_model_output_depends_on_residue_embeddings():
    torch.manual_seed(3); e=FakeFrozenEncoder(); cond=MiniSequenceConditioner(e,8,32,16); model=MiniTorsionDiffusion(single_dim=32,pair_dim=16,blocks=2,hidden_dim=32); aa=torch.zeros(1,8,dtype=torch.long); mask=torch.ones(1,8,dtype=torch.bool); cand=CandidateCondition('AAADRAAA',3,5); a=cond(sequences=[cand.sequence],aa_ids=aa,token_mask=mask,k=torch.tensor([3]),p=torch.tensor([5])); b=a._replace(single=a.single.roll(2,1)); s=TorsionState(torch.zeros(1,1,8,3),torch.ones(1,1,8,3,dtype=torch.bool),torch.zeros(1,1,8,4),torch.zeros(1,1,8,4,dtype=torch.bool)); oa=model(state_t=s,time=torch.tensor([.4]),conditioning=a,token_mask=mask,candidates=[cand]); ob=model(state_t=s,time=torch.tensor([.4]),conditioning=b,token_mask=mask,candidates=[cand]); assert not torch.allclose(oa.velocity.backbone,ob.velocity.backbone)
