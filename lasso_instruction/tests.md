# LassoDiff OpenDDE V3 测试与验收记录

状态：当前非 release V3 suite 通过；checker/surrogate/scoring release tests 待实现。

## 当前实测

命令：

```bash
python -m pytest \
  tests/data tests/unit tests/contracts tests/integration -q --maxfail=1
```

结果：`84 passed`（2026-08-01，V3.2 当前 suite）。

## 已覆盖合同

| 范围 | 证据 | 状态 |
|---|---|---|
| OpenDDE strict load/provenance | checkpoint/hash/commit/numel/frozen-eval tests + real preflight | 通过 |
| reasoning cache | identity/schema/checkpoint key/roundtrip/padding mask | 通过 |
| family split | exact-sequence/family cluster 不跨 split | 通过 |
| sequence isolation | gate API 无 k/p/ring/plug，single/pair output dependence | 通过 |
| gradient routing | gate 不更新 generator；structure 不更新 gate/reasoner | 通过 |
| candidate targets | rank-specific mapping、invalid mask、prior detach、permutation | 通过 |
| structural roles | BB/SC/NTERM/ACCEPTOR/PLUG/RING_CONTEXT 与 parent mapping | 通过 |
| dynamic geometry | SE(3) contracts、current-x_t dependence、per-block recompute | 通过 |
| diffusion/sampler | 16-block route、candidate preservation、steps-1、stability | 通过 |
| topology objective | endpoint reconstruction、formed-amide geometry、finite gradients | 通过 |
| topology checker v1 | one-oxygen chemistry、ring crossing synthetic cases、mask | 通过但非 release truth |
| checker/surrogate alignment | shared `p..tail` geometry、pre-plug isolation、double-cross rejection、rigid invariance、finite gradient | 通过 |
| truth/decoy alignment report | 6,433 real targets、unthreaded/wrong-direction decoys、hash/version/startup validation | 通过 |
| candidate scoring | hard rejection reason、candidate-preserving score components | 通过 |
| startup gate | real preflight/cache/split/alignment hash and provenance checks | 通过 |
| topology/Hamming split v4 | Hamming/LCS distance、neighbour isolation、topology balance、tamper fail-closed | 通过 |
| sequence-gate split v2 | source/mutant binding、跨 family Hamming neighbour isolation、positive topology/negative-kind strata | 通过 |
| FSDP runtime | 4 ranks、global batch 64、optimizer step、sharded checkpoint | 通过 |

## 尚未完成的 required tests

### Authoritative checker v2

- [x] non-planar ring、signed crossing count、direction 和 multiple crossing；
- [ ] open-tail closure convention；
- [ ] unthreaded/wrong-direction/wrong-plug/double-crossing decoys；
- [ ] rigid-transform invariance、小扰动稳定、边界置信度；
- [ ] 固定 truth set confusion matrix。

### Surrogate alignment

- [ ] surrogate vs checker AUROC/AUPRC/rank correlation；
- [ ] checker-labeled synthetic overfit；
- [ ] disabling surrogate hurts checker threading；
- [ ] 2,000-step pilot 中 train surrogate 与 rollout checker 同向；
- [ ] self-avoid 防止 chain-crossing shortcut。

### Scoring/physics

- [ ] hard rejection reason 完整；
- [ ] score components 与总分 manifest；
- [ ] any-clash/min-distance/bond-angle-dihedral/planarity/Ramachandran 分布；
- [ ] mean/median/p90/p95 和 length/rank/acceptor/family 分层。

### Sequence release

- [ ] verified positives 和版本化 negative strata；
- [ ] leave-family-out AUROC/AUPRC/FPR@recall/ECE/Brier/OOD false acceptance；
- [ ] ABSTAIN coverage/selective risk；
- [ ] OpenDDE-only、ESM-2-only、fusion paired ablation。

### Structure release

- [ ] checkpoint-1500/2000 与新模型相同 split/candidates/seeds/sampler steps；
- [ ] topology/threading/iso/clash/RMSD/lDDT/sampler stability 联合门禁；
- [ ] paired bootstrap CI 和预注册 non-inferiority margin。

以上 release tests 在缺少签署 truth/negative/evaluation manifest 时必须 skip/fail closed，禁止临时降低数据难度或阈值。
