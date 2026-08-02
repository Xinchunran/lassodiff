# LassoDiff Metrics Contract

文档版本：v1.0
日期：2026-08-01
适用：V2 baseline 与 OpenDDE Reasoning V3

所有指标必须携带 architecture id、checkpoint hash、split manifest hash、seed、candidate mapping、sampler steps、projection/guidance flags 和 sample count。训练 loss 下降不能替代 sequence FPR 或 rollout structure quality。

## 1. 数据与评估锁定

- Sequence/structure release 使用 topology/Hamming group-stratified split；等长序列按 normalized Hamming、变长序列按 indel-aware distance 建 neighbour clusters，完整 cluster 不拆分；同源、candidate、hard mutant/source positive 不跨 split。
- Topology allocation strata 为 acceptor type、ring length、candidate count 与 plug-gap bins；低于 `min_stratum_size` 的稀有 strata 只审计、不强行保证每个 split 都出现。
- Bootstrap 不参与数据划分，也不能修复 sequence leakage。仅在 locked test 上以 sequence-neighbour cluster 为重采样单位做 paired bootstrap；每次抽中 cluster 时保留其中全部 records/candidates，以估计模型差异置信区间。
- Structure comparison 固定相同 split、MD target mapping、seeds、candidates、sampler steps 与 projection/guidance。
- `minN/relaxN` 只与 candidate rank N 比较；不得复制单 target 到全部 M。
- validation 可用于阈值/early stopping；test 只用于最终签署。

## 2. Sequence metrics

必须报告：AUROC、AUPRC、recall、specificity、precision、FPR、FNR、coverage、selective risk、ECE、Brier score 与 OOD false acceptance。

三态统计分开报告：`LASSO_PLAUSIBLE`、`NON_LASSO`、`ABSTAIN`。不得把 ABSTAIN 算作正确 negative 来美化 accuracy；同时报告 coverage 与 covered-only performance。

Release negative sets：random、composition shuffle、acceptor/plug hard mutants、wrong topology/loop、background peptides。最低门禁：

| 指标 | 门槛 |
|---|---|
| Random acceptance rate | `<= 1%` |
| Composition-shuffle FPR | `<= 5%` |
| Hard-mutant FPR | 配置与团队签署值 |
| Overall FPR@target recall | 配置与团队签署值 |
| ECE | 配置与团队签署值 |
| OOD false acceptance | 配置与团队签署值 |

Accept/reject/OOD threshold 只能在 validation 上按目标 FPR/calibration 选择，不能用固定 0.5 或 test set 调参。

## 3. Candidate/hypothesis metrics

- acceptor top-1/top-k accuracy 与 valid-residue masked cross entropy；
- plug top-1/top-k accuracy；
- candidate NLL、Brier、entropy、top-1 probability；
- invalid candidate rate（必须通过 mask 排除，禁止 clamp）；
- candidate diversity 和 per-rank coverage；
- non-uniform teacher prior sensitivity。

V2/V3 的 posterior top-1 probability 表示 candidate marginal concentration，不等于 sequence lasso probability，也不自动等于 topology accuracy。

## 4. Structure accuracy

每个 candidate 与同编号 MD/template target 比较：

- Kabsch-aligned CA RMSD（Å）；
- Kabsch-aligned backbone RMSD（N/CA/C/O，Å）；
- CA-lDDT；
- endpoint/velocity validation loss；
- bond-length、bond-angle、plane/dihedral error；
- steric clash pair rate；
- sampler non-finite/failure rate；
- candidate diversity。

聚合必须报告 mean、median、p90/p95、sample/candidate count，并至少按长度、candidate rank、acceptor type 与 family 分层。

## 5. Lasso topology metrics

必须分别报告组成项，禁止只给一个 composite：

- NTERM--acceptor isopeptide distance/angle/plane validity；
- ring closure validity；
- ring-disk crossing/threading success；
- Gauss-link value MAE 与 link-class accuracy；
- plug-ring/tail-ring geometry；
- topology checker pass rate；
- topology diversity 与 candidate-specific success。

当前 V2 target-relative 定义：Gauss-link class 以 `-0.1/0.1` 分段为 `{-1,0,+1}`；`topology_match_rate` 要求预测与同 candidate MD target 的 link class 一致，且 isopeptide distance absolute error `<=0.5 Å`。该指标用于 V2 trend，不替代 V3 strict ring-crossing checker。

## 6. V3 structure release comparison

固定比较：V2 legacy baseline、OpenDDE frozen reasoning + adapter、以及 reasoning + lasso structural tokens + dynamic geometry。

最低 release 规则：

- threading success 必须上升；
- iso validity 不得下降；
- clash rate 不得恶化；
- RMSD/lDDT 不得显著恶化；
- sampler stability 不得下降；
- behavioral ablation 必须证明 OpenDDE pair、topology adapter、structural roles 与 dynamic geometry 各自有贡献。

统计报告应包含 paired bootstrap confidence interval；“不显著恶化”的 alpha、margin 与 bootstrap seed 必须在 locked evaluation manifest 中预注册。

## 7. 自动日志 schema

Topology-aligned V3.2 使用 `metrics_schema_version=3`。每条记录必须重复携带 architecture、OpenDDE checkpoint、split hash、seed、target policy；仅在 start record 写 provenance 不合格。训练记录额外报告 `flow/bond/iso_distance/iso_angle/iso_plane/threading` 六个 loss 分量；post-plug Gauss-link 不再作为训练 loss。

训练/validation/rollout 统一 JSONL，每行至少包含：

```json
{
  "event": "metrics",
  "architecture_id": "lassodiff_opendde_v3",
  "split": "validation_rollout",
  "step": 500,
  "checkpoint_sha256": "...",
  "split_manifest_sha256": "...",
  "sample_count": 64,
  "candidate_count": 190,
  "sampler_steps": 20,
  "projection_used": false,
  "topology_guidance_used": false,
  "metrics_schema_version": 3
}
```

NaN/Inf、sample count 变化、candidate-target mapping mismatch 或 provenance 缺失必须使评估失败，不得跳过坏样本后只报告剩余均值。

Strict isopeptide metrics 使用 formed-amide chemistry：NTERM-N、Ciso、一个 carbonyl O 和固定 7-atom schema 中的 acceptor-CA plane proxy。第二个 carboxylate oxygen 不作为 required atom。Rollout 必须同时记录绝对 `iso_distance/angle/plane` 均值及各自 validity，不能只记录 target-relative MAE。

## 8. 当前 V2 cadence

- train metrics：每 10 step；
- teacher-forced validation：每 250 step；
- 64-record 20-step rollout：每 500 step；
- final：约 400-record 40-step rollout。

V3 cadence 在正式运行前由 config 固定，并进入 run manifest。
