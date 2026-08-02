# LassoDiff OpenDDE V3 调试汇总

状态：V3 diffusion 基础设施正常；V3.2 checker/surrogate alignment 与 startup gate 已通过，长训练仍等待 2k pilot 与 release 统计。

## 已验证正常的结构主干

- OpenDDE strict checkpoint：655,791,538/655,791,538 parameters，0 missing/unexpected keys。
- frozen reasoning cache：3,122 sequences；structure phase cache-only，cache miss fatal。
- V3 medium：29,948,119 trainable parameters，`c_s/c_z/c_a=384/192/384`，16 diffusion blocks。
- Real preflight：reasoner/gate/topology/structural route 均被调用；geometry/diffusion = 16/16；single/pair 和 current `x_t` 实际影响输出。
- Structure loss 不更新 sequence gate 或 frozen reasoner；candidate prior 在 structure boundary detach。
- 4-GPU FSDP：global batch 64，bf16 params、fp32 reduction，forward/backward/optimizer/sharded checkpoint 均通过。
- Sampler stability 为 1.0；2,000-step run exit 0、final checkpoint 完整。

## 已修复的 topology=0 根因

1. 旧 checker 错误要求 formed-amide 中不存在的第二个 carboxylate oxygen，导致 strict pass 数学上不可达。
2. 旧 target policy 混用 open `minN` 与 closed `relaxN`，造成互相冲突的闭环监督。
3. velocity-only objective 没有直接约束 iso distance/angle/plane 或 topology。

修复后使用 candidate-rank-specific、chemistry-qualified targets，并从 `x_t + (1-t)v_pred` 重建 endpoint，优化 flow/bond/iso-distance/iso-angle/iso-plane/link。

## 2,000-step pilot 结果

| 指标 | step 250 | step 1,500 | step 2,000 |
|---|---:|---:|---:|
| strict topology pass | 3.73% | **5.97%** | 5.22% |
| iso distance valid | 17.91% | 32.84% | 35.07% |
| threading success | 73.88% | 38.81% | 39.55% |
| link class accuracy | 41.79% | 19.40% | 17.16% |
| CA RMSD | 8.17 Å | 6.93 Å | 6.85 Å |
| CA lDDT | 0.293 | 0.340 | 0.348 |

结论：显式 topology loss 修复了 strict pass 永久为零的问题，且没有明显牺牲 CA RMSD/lDDT；但 threading/link 指标随训练下降。训练端 link loss 从 0.198 降至 0.070，rollout link accuracy 却下降，构成直接的 surrogate/checker misalignment 证据。

## 当前主要 blocker

- 真实 checker truth report 已建立，但 wrong-plug、double-crossing、边界和 closure-convention decoy 仍需进入独立版本化 truth set。
- candidate scoring 已有 hard rejection/组件分数，但阈值与 validation manifest 尚未签署；当前 sparse-heavy-atom clash 不能解释为全原子物理质量。
- sequence gate 代码与 split 已就绪，但 verified positive/background/hard-negative manifest 仍缺失，不能启动 sequence/joint training。
- 新 alignment 尚未跑 2,000-step pilot；旧 2k checkpoint 不可用于证明新 surrogate。

## V3.2 对齐修复

- 根因：checker 检查 pre-plug `k+1..p`；旧 `gauss_linking_integral_ca` 又从 `p+1` 开始，跳过决定性的 `p→p+1` segment。两者都没有表达 candidate 的 plug-to-tail threading。
- 真实数据证据：6,433 个 chemistry-qualified rank-matched target 中，`k+1..p` 仅 219（3.40%）恰好一次 crossing，`p+1..tail` 仅 1,122（17.44%），plug-inclusive `p..tail` 为 6,211（96.55%）。
- 修复：checker 与 differentiable surrogate 共享 ordered ring centroid-fan surface、signed crossing orientation 和 `p..tail` candidate thread；double crossing 显式拒绝，Gauss link 仅保留 diagnostic。
- 初步行为审计：256 targets 对各自平移 unthreaded decoy 的 surrogate pairwise AUROC 为 0.9863，positive median 0.7966、decoy median 0；这不是完整 release truth-set 结果。
- Split：新 v4 manifest 使用 normalized Hamming/LCS neighbour single-linkage，再按 topology strata 分配完整 cluster；3,216/410/403 records，最大 cluster 400 records，manifest hash `c8a5195c18718f4a5399102069c68740c5ee81859ef6dc8dc1dd0e0861bade7e`。
- startup gate：`artifacts/v3/startup_gate_v3.json`，状态 PASS；包含 real preflight、cache manifest、v4 split hash 和 alignment report hash。
- 尚未宣称解决：需要完整 decoy truth set、validation scoring calibration、behavioral ablation 和新 2,000-step rollout。
