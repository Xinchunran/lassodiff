from types import SimpleNamespace

import torch

from lassodiff.atom_schema_lasso import ATOM_CB, ATOM_CISO, ATOM_OISO, CandidateCondition
from lassodiff.chi_geometry import build_atom14_from_rigid_groups
from lassodiff.lasso_core_decoder import decode_lasso_core
from lassodiff.residue_constants_mini import padded_atom14_names
from lassodiff.seq_encoder import seq_to_aa_ids
from lassodiff.torsion_state import TorsionState
from lassodiff.training_mini_v2 import MiniTrainingSystemV2


def _state(candidate, chi_values):
    length = len(candidate.sequence)
    backbone = torch.zeros((1, 1, length, 3))
    backbone[..., 2] = torch.pi
    backbone_mask = torch.ones_like(backbone, dtype=torch.bool)
    backbone_mask[:, :, 0, 0] = False
    backbone_mask[:, :, -1, 1:] = False
    chi = torch.zeros((1, 1, length, 4))
    chi_mask = torch.zeros_like(chi, dtype=torch.bool)
    chi[0, 0, candidate.k, :len(chi_values)] = torch.tensor(chi_values)
    chi_mask[0, 0, candidate.k, :len(chi_values)] = True
    return TorsionState(backbone, backbone_mask, chi, chi_mask)


def test_asp_ciso_is_cg_not_an_extra_carbon():
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    state = _state(candidate, [0.4, -0.7])
    core = decode_lasso_core(
        state, sequences=[candidate.sequence], candidates=[candidate],
        token_mask=torch.ones((1, len(candidate.sequence)), dtype=torch.bool),
    )[0, 0]
    assert torch.allclose(
        (core[candidate.k, ATOM_CB] - core[candidate.k, ATOM_CISO]).norm(),
        torch.tensor(1.520), atol=2e-4,
    )
    assert torch.allclose(
        (core[candidate.k, ATOM_CISO] - core[candidate.k, ATOM_OISO]).norm(),
        torch.tensor(1.240), atol=2e-4,
    )


def test_glu_atom14_uses_real_cg_cd_chain():
    candidate = CandidateCondition("AAAERAAA", 3, 5)
    state = _state(candidate, [0.2, -0.5, 1.0])
    core = decode_lasso_core(
        state, sequences=[candidate.sequence], candidates=[candidate],
        token_mask=torch.ones((1, len(candidate.sequence)), dtype=torch.bool),
    )[0, 0]
    atom14, mask = build_atom14_from_rigid_groups(
        core, seq_to_aa_ids(candidate.sequence), state.acceptor_chi[0, 0],
        state.acceptor_chi_mask[0, 0], candidate,
    )
    names = padded_atom14_names(candidate.sequence, candidate.k)[candidate.k]
    lookup = {name: index for index, name in enumerate(names) if name}
    cb, cg, cd, oe1 = (atom14[candidate.k, lookup[name]] for name in ("CB", "CG", "CD", "OE1"))
    assert mask[candidate.k, lookup["CG"]]
    assert torch.allclose((cb - cg).norm(), torch.tensor(1.520), atol=2e-4)
    assert torch.allclose((cg - cd).norm(), torch.tensor(1.520), atol=2e-4)
    assert torch.allclose((cd - oe1).norm(), torch.tensor(1.240), atol=2e-4)
    assert not torch.allclose(cg, (cb + cd) / 2)


def test_iso_closure_surrogate_targets_bond_length_not_zero():
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    endpoint = torch.zeros((1, 1, len(candidate.sequence), 7, 3))
    endpoint[0, 0, candidate.k, ATOM_CISO, 0] = 1.33
    prepared = SimpleNamespace(candidates=[candidate])
    at_target = MiniTrainingSystemV2._iso_closure_surrogate(endpoint, prepared)
    endpoint[0, 0, candidate.k, ATOM_CISO, 0] = 0.0
    collapsed = MiniTrainingSystemV2._iso_closure_surrogate(endpoint, prepared)
    assert float(at_target) < 1e-8
    assert float(collapsed) > 1.0
