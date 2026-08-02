from lassodiff.evaluation_mini_v2 import REQUIRED_SYSTEMS, validate_evaluation_report


def test_rollout_report_requires_paired_systems():
    sample = {"finite": True, "strict_valid": False, "backbone_valid": True, "formed_geometry": False,
              "crossing_count": 0, "plug_match": False, "tail_persistence": False, "clash_valid": True,
              "ca_rmsd_best_target": 1.0, "lddt_best_target": .4}
    report = {"candidate_id": "x", "seed_set": [1], "sample_count": 1,
              "systems": {name: {"samples": [sample]} for name in REQUIRED_SYSTEMS}}
    validate_evaluation_report(report)
