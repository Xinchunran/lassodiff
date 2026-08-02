from dataclasses import is_dataclass

import torch

from lassodiff.training_mini import MiniTrainingOutput, MiniTrainingSystem


def test_fsdp_root_contains_every_trainable_stage():
    system = MiniTrainingSystem(hidden_dim=24, diffusion_blocks=1)
    keys = tuple(system.state_dict())
    for prefix in ("model.", "sidechain.", "refiner.", "viability."):
        assert any(key.startswith(prefix) for key in keys)


def test_fsdp_training_output_accepts_backward_hook_rewrite():
    assert is_dataclass(MiniTrainingOutput)
    assert not MiniTrainingOutput.__dataclass_params__.frozen
    value = torch.tensor(1.0, requires_grad=True)
    output = MiniTrainingOutput(value, value, value, value, value)
    output.total = output.total * 2
    output.total.backward()
    assert value.grad.item() == 2


def test_refiner_uses_bounded_sparse_neighbors():
    system = MiniTrainingSystem(hidden_dim=24, diffusion_blocks=1)
    for block in system.refiner.blocks:
        assert 0 < block.max_neighbors < 64
