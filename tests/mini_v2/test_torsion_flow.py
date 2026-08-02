import math
import torch
from lassodiff.torsion_flow import estimate_torsion_endpoint, interpolate_torsion_state
from lassodiff.torsion_state import TorsionState


def _state(angle):
    tensor = torch.full((1,1,4,3), angle)
    return TorsionState(tensor, torch.ones_like(tensor, dtype=torch.bool), torch.zeros((1,1,4,4)), torch.zeros((1,1,4,4), dtype=torch.bool))


def test_torsion_flow_exact_endpoint_recovery():
    source, target, time = _state(-1.1), _state(1.7), torch.tensor([.37])
    flow = interpolate_torsion_state(source, target, time); recovered = estimate_torsion_endpoint(flow.state_t, flow.velocity, time)
    error = torch.atan2(torch.sin(recovered.backbone-target.backbone), torch.cos(recovered.backbone-target.backbone))
    assert float(error.abs().max()) < 1e-6


def test_torsion_flow_uses_short_path_across_pi_boundary():
    flow = interpolate_torsion_state(_state(math.radians(179)), _state(math.radians(-179)), torch.tensor([.5]))
    assert abs(float(flow.velocity.backbone.abs().max()) - math.radians(2)) < 1e-5
