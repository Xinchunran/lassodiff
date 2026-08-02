import torch
from lassodiff.atom_schema_lasso import ATOM_C, ATOM_CA, ATOM_N
from lassodiff.backbone_kinematics import build_core_from_torsions


def test_backbone_decoder_enforces_covalent_lengths():
    sequence='AAAAAA'; phi=torch.randn(6); psi=torch.randn(6); omega=torch.full((6,),torch.pi); core=build_core_from_torsions(sequence,phi,psi,omega)
    assert torch.allclose((core[:,ATOM_N]-core[:,ATOM_CA]).norm(dim=-1), torch.full((6,),1.458), atol=2e-4)
    assert torch.allclose((core[:,ATOM_CA]-core[:,ATOM_C]).norm(dim=-1), torch.full((6,),1.525), atol=2e-4)
    assert torch.allclose((core[:-1,ATOM_C]-core[1:,ATOM_N]).norm(dim=-1), torch.full((5,),1.329), atol=2e-4)


def test_backbone_decoder_is_differentiable():
    phi=torch.randn(6,requires_grad=True); psi=torch.randn(6,requires_grad=True); omega=torch.full((6,),torch.pi,requires_grad=True)
    core=build_core_from_torsions('AAAAAA',phi,psi,omega); core[:,ATOM_CA].square().sum().backward()
    assert all(g is not None and torch.isfinite(g).all() and float(g.abs().sum())>0 for g in (phi.grad,psi.grad,omega.grad))
