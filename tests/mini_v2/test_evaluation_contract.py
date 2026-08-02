from lassodiff.evaluation_mini_v2 import REQUIRED_SAMPLE_FIELDS,REQUIRED_SYSTEMS,validate_evaluation_report


def test_report_contains_paired_system_baselines():
    sample={k:False for k in REQUIRED_SAMPLE_FIELDS}; sample.update({'finite':True,'crossing_count':1,'ca_rmsd_best_target':2.1,'lddt_best_target':.65}); report={'candidate_id':'LP001:k3:p5','seed_set':[1,2,3,4],'sample_count':4,'systems':{s:{'samples':[sample.copy() for _ in range(4)]} for s in REQUIRED_SYSTEMS}}; validate_evaluation_report(report)


def test_sample_schema_is_complete():
    assert REQUIRED_SAMPLE_FIELDS


def test_missing_untrained_baseline_fails():
    sample={k:False for k in REQUIRED_SAMPLE_FIELDS}; report={'candidate_id':'x','seed_set':[1],'sample_count':1,'systems':{s:{'samples':[sample]} for s in REQUIRED_SYSTEMS if s!='untrained_model_unassisted'}}
    try: validate_evaluation_report(report)
    except ValueError: pass
    else: raise AssertionError('missing untrained baseline accepted')
