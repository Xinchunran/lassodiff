import torch
from lassodiff.checkpoint_mini_v2 import load_mini_v2_checkpoint


def test_v2_checkpoint_rejects_legacy_architecture(tmp_path):
    path=tmp_path/'legacy.pt'; torch.save({'architecture_id':'lassodiff_mini_core7','schema_version':1,'system':{}},path)
    try: load_mini_v2_checkpoint(path)
    except RuntimeError: pass
    else: raise AssertionError('legacy checkpoint accepted')
