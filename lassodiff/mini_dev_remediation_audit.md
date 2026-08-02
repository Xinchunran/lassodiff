# mini_dev remediation audit

This document is the implementation checklist for the active
`lassodiff_mini_torsion_v2` path. It is intentionally separate from the
legacy Cartesian/frame notes.

| Specification section | Active implementation / gate |
| --- | --- |
| 1 Objective | `conditioning_mini_v2`, `model_mini_v2`, `sampler_mini_v2`; headline mode is open-chain unassisted generation. |
| 2 Disposition | `atom_schema_lasso`, `mini_split`, `strict_lasso`, and `threading_mini` remain authoritative; old model/sampler/trainer are legacy. |
| 3 Grouped data | `data/mini_grouped_dataset.py`; fixed conformer dimension and masks tested in `test_grouped_dataset.py`. |
| 4 Frozen ESM | `esm_encoder_mini.py`, `conditioning_mini_v2.py`, `scripts/cache_mini_esm.py`; active-call/frozen tests present. |
| 5 Torsion flow | `torsion_state.py`, `torsion_flow.py`, `backbone_kinematics.py`; circular endpoint, boundary, length and gradient tests present. |
| 6 Dynamic geometry | `dynamic_geometry_mini.py`, `model_mini_v2.py`; rigid-transform and per-block recomputation tests present. |
| 7 Prior contract | `MiniInferenceConfig`; assisted and unassisted modes are explicit and never merged. |
| 8 Atom14 chemistry | `residue_constants_mini.py`, `chi_geometry.py`; target-only chi and bond/formed-acceptor tests present. |
| 9 Covalent refiner | `covalent_graph.py`, `MiniAtomRefinerV2`; missing graph fails in training and covalent neighbors are mandatory. |
| 10 Losses | `losses_mini_v2.py`; circular velocity and element-aware clash contracts are tested. |
| 11 Stages | `training_mini_v2.py`; stage freeze tests cover backbone, sidechain and joint modes. |
| 12 Sampler | `sampler_mini_v2.py`; Euler reference integration shares the circular parameterization. |
| 13 Evaluation | `evaluation_mini_v2.py`; paired systems and complete strict sample schema are mandatory. |
| 14 Checkpoints | `checkpoint_mini_v2.py`; legacy architecture/schema and provenance mismatch fail closed. |
| 15 Overfit | `tests/mini_v2/test_single_example_overfit.py`; slow marker is required for deterministic gate. |
| 16 Configuration | `configs/lassodiff_mini_v2.yaml`; production prior excludes single-crossing. |
| 17 Order | active files are organized by data → encoder → state → model → chemistry → stages → rollout. |
| 18 Release gates | README requires fast/slow tests and strict paired evaluation before full training. |
| 19 Rejected shortcuts | no production radial builder, hidden stitcher, predicted-chi labels, missing covalent graph, or assisted/unassisted mixing. |
| 20 Pipeline | README active pipeline ends at strict validation and best-of-K ensemble reporting. |

The existing strict evaluator is not reimplemented here. Every release sample
must pass through `lassodiff.validation.strict_lasso.strict_lasso_check`; a
construction result without that call is not a Lasso result.
