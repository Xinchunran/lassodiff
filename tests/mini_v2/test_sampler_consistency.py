import torch
from lassodiff.sampler_mini_v2 import integrate_torsion_flow
from lassodiff.torsion_flow import shortest_angular_difference
from lassodiff.torsion_state import TorsionState,TorsionVelocity


def test_oracle_velocity_integrates_exactly_to_target():
    def state(v): return TorsionState(torch.full((1,1,5,3),v),torch.ones(1,1,5,3,dtype=torch.bool),torch.full((1,1,5,4),v),torch.ones(1,1,5,4,dtype=torch.bool))
    source,target=state(-.7),state(1.1); vel=TorsionVelocity(shortest_angular_difference(target.backbone,source.backbone),shortest_angular_difference(target.acceptor_chi,source.acceptor_chi)); sampled=integrate_torsion_flow(lambda **_:vel,source,21,'euler',{}); assert float(shortest_angular_difference(sampled.backbone,target.backbone).abs().max())<1e-5 and float(shortest_angular_difference(sampled.acceptor_chi,target.acceptor_chi).abs().max())<1e-5
