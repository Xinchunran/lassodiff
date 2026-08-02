from lassodiff.evaluation_mini_v2 import REQUIRED_BASELINE_KEYS, validate_evaluation_report
from lassodiff.sampler_mini_v2 import MiniInferenceConfig


def test_unassisted_mode_does_not_use_single_crossing_prior():
    c=MiniInferenceConfig.unassisted(); assert c.prior_mode=='open_chain' and not c.projection and not c.topology_seeded


def test_assisted_mode_is_explicitly_labeled():
    c=MiniInferenceConfig.assisted(); assert c.prior_mode=='single_crossing' and c.topology_seeded


def test_evaluation_requires_all_baselines():
    r={k:{'strict_valid_rate':0.,'sample_count':8} for k in REQUIRED_BASELINE_KEYS}; validate_evaluation_report(r); del r['prior_only_unassisted']
    try: validate_evaluation_report(r)
    except ValueError: pass
    else: raise AssertionError('missing baseline accepted')
