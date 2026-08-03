import torch
from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.backbone_kinematics import build_core_from_torsions
from lassodiff.chi_geometry import build_atom14_from_rigid_groups
from lassodiff.covalent_graph import atom14_bond_geometry
from lassodiff.seq_encoder import seq_to_aa_ids


def test_atom14_builder_respects_residue_bond_lengths():
    seq='ACDEFGHIKLMNPQRSTVWY'; c=CandidateCondition(seq,2,10); L=len(seq); core=build_core_from_torsions(seq,torch.full((L,),-1.),torch.full((L,),1.),torch.full((L,),torch.pi)); a,m=build_atom14_from_rigid_groups(core,seq_to_aa_ids(seq),torch.zeros(L,4),torch.ones(L,4,dtype=torch.bool),c); assert float(atom14_bond_geometry(a[None],m[None],seq_to_aa_ids(seq)[None],[c]).length_error)<2e-2


def test_formed_acceptor_has_one_carbonyl_oxygen():
    seq='AAADRAAA'; c=CandidateCondition(seq,3,5); L=len(seq); core=build_core_from_torsions(seq,torch.zeros(L),torch.zeros(L),torch.full((L,),torch.pi)); a,m=build_atom14_from_rigid_groups(core,seq_to_aa_ids(seq),torch.zeros(L,4),torch.zeros(L,4,dtype=torch.bool),c); names=atom14_bond_geometry.atom_names_for('D',formed_acceptor=True); assert 'OD1' in names and 'OD2' not in names and torch.isfinite(a[3,m[3]]).all()
