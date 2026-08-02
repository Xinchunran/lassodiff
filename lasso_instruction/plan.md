# LassoDiff OpenDDE V3 执行计划

状态：V3 diffusion 主干可运行；V3.2 alignment/startup gate 已通过，下一步是新 alignment 2k pilot，长训练仍暂缓。

## 已完成

- [x] 固定 OpenDDE commit `f607bb3c9ff299c0627ac20f5ef8e25d716ed46f`、checkpoint SHA-256 与 strict 655,791,538-parameter load。
- [x] 建立 topology/Hamming split v4：4,029 records、3,122 unique sequences、1,453 clusters，train/val/test = 3,216/410/403；`topology_valid` = 2,674/339/353。
- [x] 完成 3,122-sequence OpenDDE reasoning cache、roundtrip manifest 和 cache-miss fail-closed。
- [x] 完成 candidate-independent gate、single/pair adapter、candidate-specific topology adapter、structural tokens、16-block dynamic geometry diffusion。
- [x] 完成 4-rank FSDP、global batch 64、bf16/fp32 mixed precision、activation checkpoint、sharded checkpoint。
- [x] 修复 candidate rank target、chemistry-qualified relax/min policy、formed-amide one-oxygen checker 与显式 topology endpoint loss。
- [x] 完成 topology-loss 2,000-step pilot；step 1,500 strict topology pass 最佳 5.97%，step 2,000 为 5.22%。
- [x] V1/toy/legacy DDP 与 V2 专属开发路径已删除；V2 run artifacts 保留为历史 baseline。
- [x] 新增 `scoring.md`，冻结 checker → surrogate → scoring 的实施边界。
- [x] 生成并验证 `artifacts/v3/startup_gate_v3.json`，trainer 强制读取 alignment report/startup gate。

## 当前不启动长训练的原因

Diffusion 基础设施正常，核心错位也已在代码中修复：训练和 checker 现共享 plug-inclusive `p..tail` segment/surface geometry。仍需先完成完整 decoy truth set、behavioral tests 和固定 2,000-step pilot；旧 run 不能证明新 loss 有效，所以当前仍不启动长训练。

## 下一轮顺序

### P0：冻结数据语义

- [ ] 与数据来源签署 `Upper_Plug_N`、`minN/relaxN` 是 candidate rank 还是 conformer ensemble。
- [ ] 将 rank/k/p/acceptor/target mapping 写入版本化 manifest。

### P1：Authoritative checker v2

- [x] 构建 non-planar centroid-fan ring surface、`p..tail` signed crossing count/class、double-crossing rejection 和 invalid reason。
- [x] 生成真实 6,433-target alignment report：target rate 96.55%、unthreaded-decoy surrogate AUC 1.0、wrong-direction class flip 99.84%，并纳入 trainer startup contract。
- [ ] 补全 plug retention、tail clearance、boundary confidence 与 small-perturbation stability。
- [x] 构建真实 target、unthreaded 与 wrong-direction alignment report；
- [ ] 补充 wrong-plug、double-crossing 与边界 decoy truth set。
- [ ] 完成刚体不变、小扰动稳定和 confusion-matrix 门禁。

### P2：Differentiable surrogate

- [x] 实现与 checker 共用 surface/thread 的 differentiable signed crossing，移除错段 checker 与跳过 `p→p+1` 的旧 Gauss-link training loss。
- [ ] 增加 signed margin、plug retention、tail clearance 和 self-avoid barrier。
- [ ] 离线验证 surrogate 与 checker label 的 AUROC/AUPRC/rank correlation。
- [ ] synthetic overfit 与 behavioral ablation 必须证明关闭 surrogate 会恶化 checker threading。

### P3：Scoring 与物理质量

- [ ] 接入 hard validity 和 candidate ranking score，保留每个组成项和 rejection reason。
- [ ] 增加 sparse-heavy-atom any-clash、minimum distance、bond/angle/dihedral、planarity、Ramachandran 和分位数报告。

### P4：训练门禁

- [ ] 先跑固定 2,000-step pilot；要求 surrogate loss 与 checker threading 同向改善。
- [ ] 锁定 checkpoint-1500/2000、相同 split/candidates/seeds/sampler steps 做 paired comparison。
- [ ] 只有 topology、threading、clash、RMSD/lDDT 联合通过才启动长训练。

### P5：Sequence gate

- [x] Sequence manifest builder 使用 source/family + Hamming/LCS neighbour cluster，并按 positive topology 与 negative kind/OOD strata 分层；manifest/hash tamper fail closed。
- [ ] 构建 verified positives 与 background/composition-matched/hard-mutant/random-OOD negatives。
- [ ] 同一 sequence、family 和 mutant/source pair 不跨 split。
- [ ] 比较 frozen OpenDDE、frozen ESM-2、frozen fusion 三路 gate；无 negatives 前不训练、不解冻 pretrained trunks。

### P6：Release

- [ ] 完成 structure behavioral ablations、locked test、paired bootstrap CI。
- [ ] 完成 sequence FPR/calibration/OOD/ABSTAIN release gates。
- [ ] threading/scoring 稳定后再单独评估 atom14/atom37 全原子升级。
