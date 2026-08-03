import torch

from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.losses_mini_v2 import (
    conformer_softmin_core_loss,
    core_clash_surrogate,
    circular_velocity_loss,
)
from lassodiff.metrics_mini_v2 import lddt_score
from lassodiff.structure_processor import canonicalize_to_root_frame


def test_target_root_frame_is_canonical_and_rigid():
    n0 = torch.tensor([4.0, -2.0, 3.0])
    ca0 = n0 + torch.tensor([0.0, 2.0, 0.0])
    c0 = n0 + torch.tensor([-1.0, 2.0, 1.0])
    coords = torch.stack((n0, ca0, c0))
    result = canonicalize_to_root_frame(coords, n0, ca0, c0)
    assert torch.allclose(result[0], torch.zeros(3), atol=1e-6)
    assert result[1, 0] > 0 and result[1, 1:].abs().max() < 1e-6
    assert result[2, 1] > 0 and result[2, 2].abs() < 1e-6


def test_core_and_atom14_can_share_one_rigid_transform():
    n0 = torch.zeros(3)
    ca0 = torch.tensor([1.0, 0.0, 0.0])
    c0 = torch.tensor([1.0, 1.0, 0.0])
    core = torch.stack((n0, ca0, c0))
    atom14 = torch.stack((n0 + torch.tensor([0.0, 0.0, 2.0]), ca0 + torch.tensor([0.0, 0.0, 2.0])))
    root_core = canonicalize_to_root_frame(core, n0, ca0, c0)
    root_atom14 = canonicalize_to_root_frame(atom14, n0, ca0, c0)
    assert torch.allclose(root_atom14[0], torch.tensor([0.0, 0.0, 2.0]), atol=1e-6)
    assert torch.allclose(root_core[1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_softmin_preserves_conformer_axis_and_exact_target_is_zero():
    prediction = torch.zeros((1, 2, 3, 1, 3))
    targets = torch.zeros((1, 2, 3, 1, 3))
    targets[:, 1] = 5.0
    masks = torch.ones((1, 2, 3, 1), dtype=torch.bool)
    conformers = torch.tensor([[True, False]])
    loss = conformer_softmin_core_loss(prediction, targets, masks, conformers)
    assert float(loss) < 1e-6


def test_softmin_is_conformer_permutation_invariant():
    prediction = torch.randn((1, 2, 3, 1, 3))
    targets = torch.randn((1, 2, 3, 1, 3))
    masks = torch.ones((1, 2, 3, 1), dtype=torch.bool)
    active = torch.tensor([[True, True]])
    left = conformer_softmin_core_loss(prediction, targets, masks, active)
    right = conformer_softmin_core_loss(prediction, targets.flip(1), masks.flip(1), active)
    assert torch.allclose(left, right)


def test_velocity_plus_two_pi_is_penalized():
    mask = torch.tensor([True])
    assert float(circular_velocity_loss(torch.tensor([0.2]), torch.tensor([0.2]), mask)) == 0.0
    assert float(circular_velocity_loss(torch.tensor([0.2 + 2 * torch.pi]), torch.tensor([0.2]), mask)) > 1.0


def test_inter_residue_clash_is_detected_but_peptide_pair_is_excluded():
    candidate = CandidateCondition("AADRAA", 2, 4)
    core = torch.zeros((1, 1, 6, 7, 3))
    for residue in range(6):
        core[0, 0, residue, 0] = torch.tensor([residue * 5.0, 0.0, 0.0])
        core[0, 0, residue, 1] = torch.tensor([residue * 5.0, 5.0, 0.0])
        core[0, 0, residue, 2] = torch.tensor([residue * 5.0, 6.0, 0.0])
        core[0, 0, residue, 3] = torch.tensor([residue * 5.0, 7.0, 0.0])
        core[0, 0, residue, 4] = torch.tensor([residue * 5.0, 5.5, 0.0])
    # Non-covalent CA-CA overlap between residues 0 and 2.
    core[0, 0, 2, 1] = core[0, 0, 0, 1] + torch.tensor([0.2, 0.0, 0.0])
    loss = core_clash_surrogate(core, [candidate], torch.ones((1, 6), dtype=torch.bool))
    assert float(loss) > 0.0


def test_true_lddt_is_not_a_rmsd_relabel():
    target = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    assert float(lddt_score(target, target)) == 1.0
    shifted = target.clone()
    shifted[-1] += torch.tensor([0.0, 0.0, 3.0])
    assert float(lddt_score(shifted, target)) < 1.0
