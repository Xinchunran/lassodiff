# frame_v1 100-step diagnostic result

- Branch: `mini_dev`
- Architecture: `lassodiff_mini_frame_v1`
- Sampler: `frame_euler_v1`
- Fold: `0`
- Seed: `17`
- World size: `4`
- Run: `runs/lassodiff-mini-frame-fold0-100-centerfix`
- Source commit used by the run: `33603a070530d10ba29dc63115d5caead679d880`
- Checkpoint: warm-started from the Cartesian Mini checkpoint at step 8000
- Evaluator: unchanged; evaluator hash guard passed locally

## Loss at step 100

Training loss:

| total | core | sidechain | refine | viability |
|---:|---:|---:|---:|---:|
| 329.4443 | 311.6595 | 59.4717 | 57.6675 | 1.2363 |

Validation loss:

| total | core | sidechain | refine | viability |
|---:|---:|---:|---:|---:|
| 347.8430 | 329.5165 | 59.3277 | 63.2726 | 1.3372 |

Frame-specific validation losses:

| translation flow | rotation flow | reactive local flow |
|---:|---:|---:|
| 58.9718 | 2.5409 | 3.5834 |

Relative to validation step 1, total validation loss decreased from `383.9894` to
`347.8430`. Translation flow decreased slightly (`59.1232` -> `58.9718`), while
rotation (`2.0639` -> `2.5409`) and reactive local flow (`1.5303` -> `3.5834`)
worsened.

## Step-100 validation metrics

| metric | value |
|---|---:|
| strict valid | 0.00% |
| formed geometry | 1.58% |
| exactly one crossing | 40.45% |
| plug match | 22.07% |
| tail persistence | 33.60% |
| backbone valid | 0.00% |
| clash valid | 6.76% |
| CA RMSD | 5.79 A |
| lDDT | 0.419 |
| frame valid | 100.00% |
| intra-residue backbone valid | 100.00% |
| C_i--N_(i+1) continuity | 23.35% |
| rotation determinant error | 2.16e-7 |

The corrected diagnostic denominators cover `48,825` valid frames and `46,605`
valid adjacent-residue pairs. The early gate therefore returns:

```text
FRAME_V1_1_CHAIN_STITCH_REQUIRED
```

## Decision

Do not continue raw `frame_v1` to a medium/full run. The residue-local frame
representation is numerically valid and fixes intra-residue geometry, but the
dominant remaining failure is inter-residue peptide continuity. Implement and
evaluate the chain stitcher before spending additional long-run compute.

The raw metrics and gate decision are stored alongside this report. Checkpoints
are intentionally not included in this result commit.
