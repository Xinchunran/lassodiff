from lassodiff.training_mini_v2 import MiniTrainingSystemV2


def _trainable(module): return {n for n,p in module.named_parameters() if p.requires_grad}


def test_backbone_stage_freezes_sidechain_and_refiner():
    s=MiniTrainingSystemV2.tiny_for_test(); s.configure_stage('backbone'); t=_trainable(s); assert any(n.startswith('backbone.') for n in t) and not any(n.startswith('sidechain.') for n in t) and not any(n.startswith('refiner.') for n in t) and not any('residue_encoder' in n for n in t)


def test_sidechain_stage_freezes_backbone():
    s=MiniTrainingSystemV2.tiny_for_test(); s.configure_stage('sidechain'); t=_trainable(s); assert any(n.startswith('sidechain.') for n in t) and not any(n.startswith('backbone.') for n in t) and not any(n.startswith('refiner.') for n in t)


def test_joint_stage_never_unfreezes_esm():
    s=MiniTrainingSystemV2.tiny_for_test(); s.configure_stage('joint'); t=_trainable(s); assert any(n.startswith('backbone.') for n in t) and any(n.startswith('sidechain.') for n in t) and any(n.startswith('refiner.') for n in t) and not any('residue_encoder' in n for n in t)
