# Mini V2 Remediation Patch

## Scope

This patch repairs the Mini V2 correctness path before any further fold training. The
authoritative production criterion remains the unchanged strict evaluator in
`lassodiff.validation.strict_lasso`. Configuration, teacher-forced loss, and legacy
preflight output are not release evidence.

## Confirmed issues

| Priority | Issue | Required correction |
| --- | --- | --- |
| P0 | PDB targets use arbitrary SE(3) coordinates while the decoder uses a canonical root frame | Canonicalize core and Atom14 with one rigid transform; preserve provenance |
| P0 | Rollout labels `1 / (1 + RMSD)` as lDDT | Use the real local-distance lDDT implementation |
| P0 | Conformer soft-min reduces the conformer axis too early | Preserve `[B, Ns, M]` and soft-min only over valid conformers |
| P0 | Velocity loss wraps tangent-space velocity errors periodically | Use direct masked SmoothL1; wrap only angular states |
| P0 | Rank-specific plug semantics are checked, but acceptor/ring semantics are not fail-closed | Validate each rank's candidate against its target PDB |
| P1 | Core clash checks only intra-residue backbone atoms | Check full-chain inter-residue pairs with covalent exclusions |
| P1 | Topology surrogate duplicates isopeptide closure | Rename it as an iso-closure surrogate; defer true crossing loss until correctness gates pass |
| P1 | Refiner trains an identity objective and drops `Ns` | Train against target chemistry and retain all samples |
| P1 | ESM has no learnable suppression gate | Add a frozen-ESM projection gate initialized near zero |

## Implemented algorithm corrections

- Canonicalized core and Atom14 targets with one rigid root-frame transform.
- Corrected Asp/Glu reactive kinematics: Asp `CISO=CG`; Glu uses a real
  `CB-CG-CD(CISO)-OE1(OISO)` chain. Acceptor chi now moves CISO/OISO.
- Added an offline, content-addressed decoder-manifold target fit. Production
  dataset construction validates the manifest hash, raw strict validity,
  candidate identity and fit convergence, and fails closed on any mismatch.
- Preserved candidate identity `(record_id, sequence, k, p)` and conformer axis;
  rank-specific PDB acceptor identity is audited instead of inferred silently.
- Corrected conformer soft-min reduction, tangent velocity loss, circular
  endpoint angle loss, true lDDT, full-chain core clash and evaluator routing.
- Added a staged endpoint warm-up so the large coordinate loss cannot establish
  extra angular winding before the shortest-path flow field.
- Heun no longer queries the ambiguous `t=1` velocity; the last step uses the
  identifiable `t<1` velocity. A low-frequency sampler-in-the-loop endpoint
  objective now exposes training to the path actually used at inference.
- Full-Atom14 sampling now routes sidechain prediction and the mandatory
  covalent graph through the refiner before strict evaluation.

The strict evaluator source and thresholds were not changed by this patch.

## Acceptance gates

The following must be measured, not inferred:

1. Real-PDB extract/decode canonical direct CA RMSD `< 0.75 A`.
2. Oracle flow rollout true lDDT `> 0.95`.
3. Single-candidate teacher-forced true lDDT `> 0.85`.
4. Single-candidate rollout CA RMSD `< 1.5 A` and true lDDT `> 0.70`.
5. Finite rate `100%` and backbone bond validity `100%` for the single-candidate gate.
6. Rank 1/2/3 candidates are reported separately with RMSD, true lDDT, and strict results.

Any failed gate is reported as `FAIL`; ideal values must never be substituted.

## Metrics contract

`artifacts/mini_v2/remediation_metrics.json` and `.jsonl` must include candidate
identity, rank, `k`, `p`, loop size, target source, prior mode, sampler settings,
canonical and aligned CA RMSD, true lDDT, finite/backbone/chemistry metrics, strict
validity, and strict rejection reasons. The report must include architecture/schema,
source commit, dataset mapping hash, canonicalization version, and strict checker
identity.

## Current measured status

Fast V2 tests pass: `54 passed, 1 deselected`. The slow gate is intentionally
not skipped and is currently `FAIL`, so no fold training is authorized.

The real `LP_14506` decoder-fit audit at source commit
`86684e39484802c30fb3e7aaae05db1c442b8c6d` reports:

| rank | k | p | fit-to-raw CA RMSD (A) | true lDDT | core strict | full Atom14 strict |
| ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 1 | 8 | 13 | 0.3655 | 0.9667 | PASS | FAIL: `severe_clash` |
| 2 | 8 | 12 | 0.3411 | 0.9625 | PASS | FAIL: `severe_clash` |
| 3 | 8 | 11 | 0.4020 | 0.9521 | PASS | FAIL: `severe_clash` |

This proves that target/decoder gauge and reactive-core geometry are repaired,
but it does not prove full-heavy-atom generation. The top-level remediation
report is correctly `FAIL` because strict evaluates the final Atom14 structure.

### Single-example experiments retained as failures

All runs used the same 1,000-step real optimizer route, fixed open-chain source,
four rollout samples and 60 sampler steps. No result below is a release pass.

| change | best CA RMSD (A) | true lDDT | outcome |
| --- | ---: | ---: | --- |
| original corrected baseline | 5.08 | 0.657 | FAIL |
| smooth time embedding only | 5.66 | 0.833 | FAIL |
| isolate flow/endpoint and circular endpoint loss | 2.34 | 0.833 | FAIL; best RMSD so far |
| condition-only velocity skip | 4.40 | 0.926 | FAIL; local geometry improved, trajectory drifted |
| strong one-step recovery loss | 7.18 | 0.833 | FAIL; reverted |
| weak one-step recovery loss | 5.14 | 0.806 | FAIL; reverted |
| larger 64-dimensional tiny model | 4.75 | 0.713 | FAIL; reverted |

The current sampler-in-the-loop version reduces pointwise backbone flow to about
`0.078`, but its scheduled final total is about `0.983`; it therefore fails the
95% loss-reduction assertion before rollout metrics are accepted. This is the
active blocker, together with the untrained full-Atom14 clash failure above.

The V2 preflight may validate route identity and forward/backward behavior, but
it cannot override either failed gate. Five-fold and fold-0 production training
remain prohibited.
