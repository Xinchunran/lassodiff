import inspect
import torch
from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.backbone_kinematics import build_core_from_torsions
from lassodiff.chi_geometry import build_atom14_from_rigid_groups, extract_chi_angles, rotamer_classes_from_target_chi
from lassodiff.seq_encoder import seq_to_aa_ids


def test_chi_extraction_recovers_constructed_target():
    c=CandidateCondition('AKADRAAA',3,5); L=8; core=build_core_from_torsions(c.sequence,torch.full((L,),-1.),torch.full((L,),1.2),torch.full((L,),torch.pi)); chi=torch.zeros(L,4); mask=torch.zeros(L,4,dtype=torch.bool); chi[1]=torch.tensor([.6,-1.2,2.,-.4]); mask[1]=True; atoms,am=build_atom14_from_rigid_groups(core,seq_to_aa_ids(c.sequence),chi,mask,c); t=extract_chi_angles(atoms[None,None],am[None,None],seq_to_aa_ids(c.sequence)[None],[c]); e=torch.atan2(torch.sin(t.angles[0,0,1]-chi[1]),torch.cos(t.angles[0,0,1]-chi[1])); assert float(e[t.masks[0,0,1]].abs().max())<1e-4


def test_rotamer_targets_do_not_accept_predicted_chi():
    assert {'prediction','predicted','predicted_chi','logits'}.isdisjoint(inspect.signature(rotamer_classes_from_target_chi).parameters)
