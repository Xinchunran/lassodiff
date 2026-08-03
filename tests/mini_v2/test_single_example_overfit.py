import pytest
import torch

from lassodiff.training_mini_v2 import MiniTrainingSystemV2, make_synthetic_overfit_batch


@pytest.mark.slow
def test_single_example_can_be_overfit():
    torch.manual_seed(11)
    system = MiniTrainingSystemV2.tiny_for_test(); system.configure_stage("backbone"); system.train()
    batch = make_synthetic_overfit_batch("AAADRAAA", 3, 5, batch_size=1)
    optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad], lr=2e-3)
    initial = None
    for step in range(1, 1001):
        output = system.forward_backbone_stage(batch, global_step=step)
        if initial is None: initial = float(output.total.detach())
        optimizer.zero_grad(set_to_none=True); output.total.backward(); optimizer.step()
    assert float(output.total.detach()) < initial * .05
    rollout = system.rollout_for_test(batch, samples=4, steps=60)
    assert rollout.finite_rate == 1.0
    assert rollout.backbone_bond_valid_rate == 1.0
    # The real rollout gate is deliberately separate from the teacher-forced
    # endpoint gate: canonical decoding must be finite and chemically valid,
    # while the initial model gate allows a 1.5 A rollout envelope.
    assert rollout.best_ca_rmsd < 1.5
    assert rollout.best_lddt > .70
