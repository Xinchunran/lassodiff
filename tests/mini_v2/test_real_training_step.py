import torch
from lassodiff.training_mini_v2 import MiniTrainingSystemV2, make_synthetic_overfit_batch


def test_real_training_step_updates_backbone():
    system = MiniTrainingSystemV2.tiny_for_test(); batch = make_synthetic_overfit_batch("AAADRAAA", 3, 5)
    optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad], lr=1e-3)
    before = {n: p.detach().clone() for n, p in system.named_parameters() if p.requires_grad}
    output = system.forward_backbone_stage(batch, generator=torch.Generator().manual_seed(11), global_step=1)
    assert torch.isfinite(output.total)
    optimizer.zero_grad(); output.total.backward(); optimizer.step()
    assert any(not torch.allclose(before[n], p) for n, p in system.named_parameters() if n in before)
