import pytest
import torch

from lassodiff.conditioning_mini_v2 import MiniConditioning


@pytest.fixture
def fake_conditioning():
    return MiniConditioning(torch.randn((1, 8, 32)), torch.randn((1, 8, 8, 16)), torch.ones((1, 8, 8), dtype=torch.bool))
