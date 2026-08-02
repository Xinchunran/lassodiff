import torch
from lassodiff.atom_schema_lasso import ATOM_CISO, ATOM_OISO, CandidateCondition
from lassodiff.lasso_core_decoder import decode_lasso_core
from lassodiff.torsion_state import TorsionState


def test_acceptor_chi_changes_ciso_oiso():
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    base = TorsionState(torch.zeros(1, 1, 8, 3), torch.ones(1, 1, 8, 3, dtype=torch.bool),
                        torch.zeros(1, 1, 8, 4), torch.zeros(1, 1, 8, 4, dtype=torch.bool))
    base.acceptor_chi_mask[0, 0, 3, :2] = True
    changed = base.clone(); changed.acceptor_chi[0, 0, 3, :2] = torch.tensor([1.0, -.8])
    mask = torch.ones(1, 8, dtype=torch.bool)
    a = decode_lasso_core(base, sequences=[candidate.sequence], candidates=[candidate], token_mask=mask)
    b = decode_lasso_core(changed, sequences=[candidate.sequence], candidates=[candidate], token_mask=mask)
    assert not torch.allclose(a[..., ATOM_CISO, :], b[..., ATOM_CISO, :])
    assert not torch.allclose(a[..., ATOM_OISO, :], b[..., ATOM_OISO, :])
