import torch

from lassodiff.atom_refiner import MiniAtomRefiner
from lassodiff.atom_schema_lasso import ATOM_CISO, ATOM_OISO, CandidateCondition
from lassodiff.peptide_prior import sample_peptide_prior
from lassodiff.seq_encoder import seq_to_aa_ids
from lassodiff.sidechain_builder import atom14_mask, atom14_names, build_atom14, symmetry_aware_coordinate_loss


def test_atom14_masks_match_residue_types():
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    mask = atom14_mask(candidate.sequence, candidate)
    assert atom14_mask("AGDRAAA", CandidateCondition("AGDRAAA", 2, 4))[1].sum() == 4
    names = atom14_names(candidate.sequence, candidate)[3]
    assert "OD1" in names and "OD2" not in names


def test_sidechain_builder_preserves_chirality():
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    prior = sample_peptide_prior(candidate, mode="open_chain", generator=torch.Generator().manual_seed(8), noise_scale=0)
    built, mask = build_atom14(prior.coordinates, candidate)
    names = atom14_names(candidate.sequence, candidate)
    for residue, aa in enumerate(candidate.sequence):
        if aa != "G":
            assert torch.allclose(built[residue, names[residue].index("CB")], prior.coordinates[residue, 4])
    assert mask.any()


def test_symmetric_atoms_use_minimum_permutation_loss():
    candidate = CandidateCondition("AFADRAAA", 3, 5)
    prior = sample_peptide_prior(candidate, mode="open_chain", generator=torch.Generator().manual_seed(9), noise_scale=0)
    predicted, mask = build_atom14(prior.coordinates, candidate)
    target = predicted.clone()
    names = atom14_names(candidate.sequence, candidate)[1]
    left, right = names.index("CD1"), names.index("CD2")
    target[1, [left, right]] = target[1, [right, left]]
    assert float(symmetry_aware_coordinate_loss(predicted, target, candidate.sequence, mask, candidate)) < 1e-6


def test_iso_acceptor_is_not_treated_as_normal_asp_glu():
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    prior = sample_peptide_prior(candidate, mode="single_crossing", noise_scale=0)
    built, _mask = build_atom14(prior.coordinates, candidate)
    names = atom14_names(candidate.sequence, candidate)[3]
    assert "OD2" not in names
    assert torch.allclose(built[3, names.index("CG")], prior.coordinates[3, ATOM_CISO])
    assert torch.allclose(built[3, names.index("OD1")], prior.coordinates[3, ATOM_OISO])


def test_atom_refiner_is_bounded_and_equivariant():
    torch.manual_seed(10)
    candidate = CandidateCondition("AAADRAAA", 3, 5)
    prior = sample_peptide_prior(candidate, mode="single_crossing", noise_scale=0)
    atoms, mask = build_atom14(prior.coordinates, candidate)
    aa = seq_to_aa_ids(candidate.sequence)[None]
    model = MiniAtomRefiner(hidden_dim=24, layers=2, max_displacement=.5)
    output = model(atoms[None], aa, mask[None])
    assert float((output - atoms[None]).norm(dim=-1)[mask[None]].max()) <= .5001
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotated = model((atoms @ rotation)[None], aa, mask[None])
    assert torch.allclose(rotated, output @ rotation, atol=2e-5, rtol=2e-5)
