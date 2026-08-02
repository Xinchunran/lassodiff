import torch
from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.atom_refiner import MiniAtomRefinerV2, select_refiner_neighbors
from lassodiff.covalent_graph import build_atom14_covalent_graph
from lassodiff.seq_encoder import seq_to_aa_ids


def test_graph_contains_peptide_and_isopeptide_bonds():
    c=CandidateCondition('AAADRAAA',3,5); ids=seq_to_aa_ids(c.sequence)[None]; g=build_atom14_covalent_graph(ids,torch.ones_like(ids,dtype=torch.bool),[c]); assert g.peptide_edge_count==7 and g.isopeptide_edge_count==1 and bool(g.adjacency.any())


def test_covalent_neighbors_are_never_dropped():
    x=torch.zeros(1,6,3); x[0,:,0]=torch.tensor([0.,100.,.1,.2,.3,.4]); m=torch.ones(1,6,dtype=torch.bool); c=torch.zeros(1,6,6,dtype=torch.bool); c[0,0,1]=c[0,1,0]=True; n=select_refiner_neighbors(x,m,c,3); assert 1 in n[0,0].tolist()


def test_training_refiner_requires_covalent_graph():
    r=MiniAtomRefinerV2(hidden_dim=24,layers=1,max_neighbors=8); r.train()
    try: r(torch.zeros(1,2,14,3),torch.zeros(1,2,dtype=torch.long),torch.ones(1,2,14,dtype=torch.bool),covalent_adjacency=None,bond_type=None)
    except ValueError: pass
    else: raise AssertionError('missing graph accepted')
