import torch
from lassodiff.backbone_kinematics import build_core_from_torsions, extract_backbone_torsions
from lassodiff.torsion_flow import shortest_angular_difference


def test_backbone_decoder_extractor_roundtrip():
    sequence = "ACDEFGHIK"; L = len(sequence)
    phi = torch.tensor([0., -1.2, -.8, -1.1, -.9, -1.3, -.7, -1., -.6])
    psi = torch.tensor([1., .8, 1.2, .5, 1.1, .7, 1.3, .9, 0.])
    omega = torch.tensor([torch.pi] * (L - 1) + [0.])
    recovered, mask = extract_backbone_torsions(build_core_from_torsions(sequence, phi, psi, omega))
    assert torch.allclose(shortest_angular_difference(recovered[1:, 0], phi[1:]), torch.zeros(L - 1), atol=1e-4)
    assert torch.allclose(shortest_angular_difference(recovered[:-1, 1], psi[:-1]), torch.zeros(L - 1), atol=1e-4)
    assert torch.allclose(shortest_angular_difference(recovered[:-1, 2], omega[:-1]), torch.zeros(L - 1), atol=1e-4)
