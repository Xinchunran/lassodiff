# LassoDiff v3.2.0

LassoDiff 是面向套索肽（lasso peptide）的 sequence screening、candidate-specific topology reasoning 与三维结构生成仓库。当前唯一开发主线是 `lassodiff_opendde_v3`：冻结 OpenDDE residue reasoning，并分别服务于 sequence-level `LASSO / NON_LASSO / ABSTAIN` 判别与 candidate-specific geometry-aware diffusion。V1/toy 路径已退役；V2 只作为冻结的历史 baseline，不再作为开发入口。

文档版本：3.2.0

更新日期：2026-08-01

V3 状态：Implementation contract / test-first runbook

## Agent 启动协议（MUST）

任何 agent 在每次开始本仓库任务时，必须先完整阅读本 README，再执行读取状态以外的命令或修改。涉及 V3 reasoning、训练、筛选或评估时，还必须按顺序阅读：

1. [`reasoning.md`](reasoning.md)：OpenDDE reasoning 架构、禁止项、阶段与测试合同；
2. [`metrics.md`](metrics.md)：sequence/structure 指标定义、数据切分与 release gate；
3. [`scoring.md`](scoring.md)：authoritative threading checker、可微 surrogate 与 candidate scoring 合同；
4. [`lasso_instruction/plan.md`](lasso_instruction/plan.md)、[`debug.md`](lasso_instruction/debug.md)、[`tests.md`](lasso_instruction/tests.md)：当前执行状态与历史证据。

不得仅凭类名、配置、日志字符串或 state dict 宣称功能完成。所有 `MUST` 均需 route、output-dependence、gradient、optimizer、checkpoint manifest 和 behavioral test 证据。OpenDDE checkpoint/commit/schema 不匹配时必须 fail closed，禁止回退 ESM-only、legacy Pairformer 或旧 score head。

仓库根目录 [`AGENTS.md`](AGENTS.md) 同步声明该协议，供自动化 agent loader 强制执行。

## 当前状态

| 路径 | 状态 | 说明 |
|---|---|---|
| V2 medium baseline | 已冻结 | 20,000 step artifact 保留用于历史比较；源码入口不再属于开发主线 |
| V2 small baseline | 已冻结 | step 1,620 停止，checkpoint/log 保留 |
| OpenDDE V3 velocity-only baseline | 20,000 step 已完成、判定失败 | strict topology pass 始终 0；artifact 全部保留，不作为 release model |
| OpenDDE V3 topology-loss 2k | 历史诊断完成 | exit 0；旧 checker 错看 pre-plug `k+1..p`，旧 loss 又跳过决定性的 `p→p+1`，两者均未表达 plug-to-tail threading；该 run 不再续训 |
| V3.2 threading alignment | 代码与离线数据审计通过 | checker/surrogate 共享非平面 centroid-fan surface 与 `p..tail` thread；6,211/6,433（96.55%）真实 targets 恰好一次 crossing；新 2k pilot 尚未运行 |
| V3.2 startup gate | PASS | real preflight、cache manifest、topology/Hamming split、6,433-target alignment report 已通过 hash/provenance 校验；可启动新 2k pilot |

历史 V2 medium artifact 位于 [`runs/lassodiff-v2-medium-20260801/`](runs/lassodiff-v2-medium-20260801/)，只用于结果追溯，不再提供 V2 训练入口。

V3 medium 状态位于 [`runs/lassodiff-v3-structure-medium-20260801/`](runs/lassodiff-v3-structure-medium-20260801/)。`launcher.log`、`metrics.jsonl`、`train_status.json`、config/preflight/cache/split/source manifests 与每 500-step FSDP checkpoints 均保存在该目录。

修复后的 2,000-step 任务位于 [`runs/lassodiff-v3-topology-2k-20260801/`](runs/lassodiff-v3-topology-2k-20260801/)，已成功结束并保存 checkpoint-250 到 checkpoint-2000 及 final。V3 structure target 只使用 chemistry-qualified candidate labels：优先同 rank 的有效 `relaxN`，没有 relax 时才使用通过闭环化学检查的 `minN`；无可靠 target 的 candidate 显式 mask，禁止把开环模板当闭环监督。

源码范围已经收敛到 V3：V1/toy/legacy DDP 与 V2 专属模型、训练入口、配置和专属测试已删除。历史 run artifacts 不删除；V3 仍依赖的通用 candidate/data/geometry/evaluation 数学组件保留。

## V3 总体数据流

```text
Sequence
  -> frozen OpenDDE residue reasoning -> s_res [B,L,Cs], z_res [B,L,L,Cz]
       |-> candidate-independent sequence gate -> LASSO / NON_LASSO / ABSTAIN
       `-> reasoning adapter -> candidate topology(k_j,p_j,acceptor_j)
            -> lasso structural tokens -> per-block dynamic geometry diffusion
            -> candidate structures -> strict topology checker
```

核心边界：sequence gate 的 API 不得接收 `k/p/acceptor/ring/plug/closure edge`；screening 不运行 diffusion、不做 hard projection/guidance；结构路径必须依赖 OpenDDE `pair` state，并为每个 candidate 使用独立坐标、atom mask 和 target。

## 数据与当前结构表示

- 原始 metadata：`${LASSOPRED_ROOT}/lassopred.data.json`
- 原始 MD/template PDB：仓库 `structure/<LP_ID>/min[1-3].pdb|relax[1-3].pdb`
- LMDB：`data/lassopred.lmdb`，4,029 records
- ESM-2 650M cache：`data/esm2_t33_650M_UR50D.lmdb`，3,122 unique sequences
- V3 topology/Hamming split v4：3,216 / 410 / 403；3,122 unique sequences、1,453 neighbour clusters；结构训练经 `topology_valid` 过滤后为 2,674 / 339 / 353
- Split 先以等长 normalized Hamming、变长 LCS distance 做 single-linkage 隔离，再按 acceptor/ring length/candidate count/plug-gap topology strata 做 group allocation。Bootstrap 不用于构造 split；它只用于 locked test 上的 paired confidence interval。

`Upper_Plug_1/2/3` 定义 candidate rank 1/2/3；`min1/relax1`、`min2/relax2`、`min3/relax3` 分别只监督同 rank candidate。它们当前不是“同一 topology 的三个无条件 conformation”。候选维 `M` 在 candidate objective 前不得消失；若要生成同 candidate 的构象 ensemble，应固定 candidate 后使用多个 diffusion noise seeds。

当前不是全原子生成：坐标 schema 是每个 residue 的 `N/CA/C/O`，加 acceptor residue 的 `Ciso/O1`。因此 clash 只能称为 sparse-heavy-atom clash，不能据此宣称全原子物理有效。

## 下一轮 TODO（按执行顺序）

- [ ] 与数据来源再次签署 `Upper_Plug_N` 和 `minN/relaxN` 的 candidate/conformation 语义；在 manifest 中冻结，不凭文件名继续推断。
- [x] 在已统一的 `p..tail` 非平面 surface 定义上补齐 checker 的 plug retention、tail clearance、扰动稳定性字段；
- [ ] 补全 wrong-plug、double-crossing、边界和 closure-convention 版本化 decoy truth set。
- [x] 移除错段 checker/旧 Gauss-link training loss，接入与 checker 共用几何对象的 differentiable signed-crossing surrogate；真实 target hard-check 一致率 96.55%。
- [ ] 完成 surrogate AUROC/AUPRC/rank correlation、synthetic overfit、behavioral ablation 和新 2,000-step rollout；未通过前仍不启动长训练。
- [x] 将 checker hard validity 与各组成项接入 candidate scoring；日志保留 rejection reason、geometry/clash/threading 组件，且不回写 sequence classification。
- [ ] 在 locked validation 上校准 scoring 权重与 hard thresholds。
- [ ] 完善 sparse-heavy-atom clash 与主链分布报告：any-clash、minimum distance、bond/angle/dihedral、planarity、Ramachandran、分位数和分层统计。
- [x] 结构数据落地 topology/Hamming group-stratified split v4，并由 cluster/topology/hash fail-closed tests 约束。
- [x] `build_sequence_gate_dataset.py` 移除随机 group shuffle，改为 source/family + Hamming/LCS neighbour hard grouping，并联合分层 positive topology 与 negative kind/OOD strata。
- [ ] 构建 verified lasso-positive manifest 与 background/composition-shuffle/hard-mutant/verified wrong-topology sequence controls/random-OOD negatives；sequence gate split 复用相同 neighbour isolation，并额外绑定 mutant/source group。仅给同一 sequence 换错误 `k/p` 不是 sequence negative，只能监督 candidate hypothesis/structure。
- [ ] 在完全相同 split 上做 `OpenDDE gate`、`frozen ESM-2 gate`、`OpenDDE+ESM fusion` 三路 ablation；第一轮两个 pretrained trunks 均冻结，只训练小型 adapter/head。
- [ ] 在 negatives、FPR/calibration 与 leave-family-out 证据完成前，不启动 sequence joint training，不把 3,122 条 LassoPred 序列全部冒充 verified positives。
- [ ] 锁定 checkpoint-1500/2000、相同 candidate/seeds/sampler steps，完成 paired bootstrap 的 structure release comparison。
- [ ] threading/scoring 稳定后再单独决定是否从 sparse 7-slot 升级 atom14/atom37；禁止与 surrogate 重构同时改动。
- [ ] 将仍被 V3 复用但历史命名带 `v2` 的通用评估函数重命名为版本中性模块，并保持 checkpoint/metric schema provenance。

## OpenDDE 固定版本

本地运行前需将 OpenDDE runtime 根目录通过环境变量注入，仓库不保存个人机器路径：

```bash
export OPENDDE_RUNTIME_ROOT=/path/to/opendde_data
```

- 官方仓库：<https://github.com/aurekaresearch/OpenDDE>
- 审计并固定 commit：`f607bb3c9ff299c0627ac20f5ef8e25d716ed46f`
- 本机 checkpoint：`opendde.pt`
- checkpoint SHA-256：`7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc`
- 许可证：Apache-2.0；复制/修改源码必须保留 SPDX/header 并记录本地修改

OpenDDE 是 preview，公开 CLI 不保证暴露稳定的中间 reasoning state。V3 必须在固定 fork 中提供版本化 `forward_reasoning()`，不得依赖 floating `main`、匿名 module index 或静默 hook fallback。

## 常用命令

V3 非 release tests：

```bash
python -m pytest \
  tests/data tests/unit tests/contracts tests/integration -q --maxfail=1
```

已完成 topology-loss run：

```bash
jq -c 'select(.split=="validation_rollout" or .event=="complete")' \
  runs/lassodiff-v3-topology-2k-20260801/metrics.jsonl
```

V3 的训练命令在 `reasoning.md` 所列 P0--P6 门禁和 V3 preflight 全部通过前不得运行。

V3 全量 reasoning cache（当前由同名 systemd user service 持久运行）：

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/cache_opendde_reasoning.py \
  --config configs/lassodiff_opendde_v3.yaml \
  --dataset data/lassopred.lmdb \
  --log-dir runs/lassodiff-v3-opendde-cache-20260801 --device cuda
```

V3 正式 structure phase 固定为 4-rank FSDP、每卡 batch 16、global batch 64；入口会拒绝 cache miss、旧 record split 和未记录真实 reasoner route 的 preflight：

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/train_structure_v3.py \
  --config configs/lassodiff_opendde_v3.yaml \
  --dataset data/lassopred.lmdb \
  --split data/lassopred.lmdb/split_topology_hamming_v4.json \
  --preflight artifacts/v3/preflight.real.json \
  --cache-manifest artifacts/v3/cache_manifest.json \
  --threading-report artifacts/v3/threading_alignment_v3.json \
  --startup-gate artifacts/v3/startup_gate_v3.json \
  --run-dir runs/lassodiff-v3-structure-medium-20260801 \
  --batch-size 16 --max-steps 20000
```

Sequence gate 和 joint trainer 已提供，但只有在 verified lasso positive 与版本化 background/hard-negative manifest 就绪后才允许启动。LassoPred candidate/MD template 只能监督候选结构，不能被当成 sequence positive 标签。

## 版本政策

- `VERSION` 是仓库文档/架构主版本；当前为 `3.2.0`。
- V2 checkpoint manifest 的 schema 保持 2；V3 使用 `architecture_id=lassodiff_opendde_v3`、schema 3。
- V1/V2 checkpoint 不得通过 `strict=False` 直接加载到 V3。
- config、source commit、OpenDDE commit/checkpoint SHA、feature schema、split manifest 和 metrics schema 必须一同归档。

## Acknowledgements

LassoDiff 参考并复用 Protenix、ml-simplefold 与 OpenDDE 的公开工程和建模思想。OpenDDE 来源代码受 Apache-2.0 约束；任何 vendored/modified code 必须保留原始许可声明。
