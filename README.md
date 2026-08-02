# LassoDiff Mini

`mini_dev` 是不依赖同源模板、OpenDDE 权重或全原子高斯初始化的小模型开发线。它直接替代旧的“template + classifier 选择 iso/plug”生成路径：

```text
sequence + candidate(k,p)
  -> programmatic peptide prior
  -> core-7 equivariant diffusion
  -> rotamer/chi sidechain builder
  -> bounded 4-layer sparse-kNN all-heavy-atom refiner
  -> strict chemistry/topology checker
```

当前版本：`4.0.0-mini.1`。架构 preflight 与 mini 单测已通过；仓库尚未提供训练完成的 mini checkpoint，因此不能把程序化 seed 或随机权重输出称为已训练生成结果。

## 开发边界

- Mini 的唯一坐标 schema 是 `N/CA/C/O/CB/CISO/OISO`。
- `ASP_ISO` 使用 `CISO=CG, OISO=OD1`；`GLU_ISO` 使用 `CISO=CD, OISO=OE1`。
- formed acceptor 只有一个 carbonyl oxygen；不存在的 atom 必须由 mask 表示，禁止用零坐标冒充。
- 每个 `(sequence,k,p,target-rank)` 是独立训练样本；非法 candidate 直接失败，不做 `clamp()`。
- prior 只使用 sequence、`k/p` 和通用肽链几何，不读取模板坐标。
- screening 固定使用 open-chain prior，并关闭 projection/topology guidance。
- strict checker 是最终真值；construction 成功率不能替代 sequence feasibility。

原 OpenDDE V3 文件目前只作为历史/活跃实验兼容代码保留，不再是本分支默认入口。README 记录的 V3.3 任务在分叉时仍有活跃 run，因此本次没有删除或移动它可能继续使用的源码、配置、日志和 checkpoint。待该 run 结束后可以单独清理 legacy 文件。

## 环境与首次检查

每次任务仍须先完整阅读本 README。不要下载数据、启动训练或控制进程，直到 startup preflight 通过。

```bash
python -m scripts.verify_mini_startup \
  --source-split data/lassopred.lmdb/split_topology_hamming_v4.json \
  --cv-split data/mini_cv5_locked_test_v1.json \
  --output artifacts/mini/preflight.json

python -m pytest tests/mini -q
```

preflight 会行为验证：ASX 单氧 chemistry、无模板 prior、0/1/2 crossing fixtures、current-coordinate dependence、screening 无 projection/guidance、侧链完整性和 refiner 最大位移。任一失败都不得训练。

## 数据

Mini 不读取旧 schema-1 LMDB 的 7 槽坐标，因为该 LMDB 是 `N/CA/C/O/CISO/O1/O2`，与新 core-7 不兼容。训练时从原始 metadata 与 rank-matched PDB 重新解析：

```text
Upper_Plug_1 <-> relax1/min1
Upper_Plug_2 <-> relax2/min2
Upper_Plug_3 <-> relax3/min3
```

同 rank 先验证 `relaxN`，缺失或非有限/化学不合格时再验证 `minN`；两者都失败的 example 显式计入 rejection，不会在训练中途随机报错。PDB 会去氢、规范化 `ASX/GLX`、生成 core/Atom14 mask，并保持 candidate 维度。所有 rank 必须得到相同的 qualified count、rejection count 与 mapping SHA-256，否则 FSDP 在创建 run 目录前失败。

数据拆分继承 V3 的 topology/Hamming/LCS neighbour-cluster 合同。V3 的 403-record test 集合永久 hold out，不参与训练、交叉验证或 early stopping；原 train+validation 合并为 development pool，以完整 neighbour cluster 为单位做 topology-stratified 5-fold。每个 development record 恰好作为一次 validation，同一 sequence/近邻 cluster 的全部 candidates 不得跨 fold。当前 Mini PDB 映射缺少的 7 个 source records 保留原归属并在 manifest 中显式标记，禁止通过移动 test 或拆 cluster 补齐比例。锁定 manifest 是 `data/mini_cv5_locked_test_v1.json`。

需要重建该 manifest 时必须从锁定 V3 split 和当前 Mini PDB 映射生成，不得手工编辑：

```bash
python -m scripts.build_mini_cv_split \
  --source-split data/lassopred.lmdb/split_topology_hamming_v4.json \
  --metadata /path/to/lassopred.data.json --structure-root /path/to/structure \
  --output data/mini_cv5_locked_test_v1.json --seed 17
```

## 训练

正式训练使用单机 4-GPU FSDP full-shard；`--batch-size` 是每卡 batch，下面的 global batch 为 16。它不会自动下载数据或停止现有任务：

```bash
/home/ranx/miniconda3/envs/DeltaCata/bin/torchrun \
  --standalone --nproc-per-node=4 \
  -m scripts.train_mini \
  --metadata /path/to/lassopred.data.json \
  --structure-root /path/to/structure \
  --preflight artifacts/mini/preflight.json \
  --split data/mini_cv5_locked_test_v1.json \
  --source-split data/lassopred.lmdb/split_topology_hamming_v4.json \
  --fold 0 --run-dir runs/lassodiff-mini-cv/fold-0 \
  --batch-size 4 \
  --steps 20000 --log-every 10 --save-every 500 --validate-every 500
```

正式 5-fold 训练由顺序 launcher 在同一组 4 GPU 上逐 fold 执行，避免五个模型争抢显存：

```bash
python -m scripts.train_mini_cv \
  --metadata /path/to/lassopred.data.json --structure-root /path/to/structure \
  --preflight artifacts/mini/preflight.json \
  --split data/mini_cv5_locked_test_v1.json \
  --source-split data/lassopred.lmdb/split_topology_hamming_v4.json \
  --run-root runs/lassodiff-mini-cv5 --steps 20000
```

每个 fold 都使用其余四 folds 训练并在唯一 held-out fold 上记录 validation loss；test 不会被 loader 打开。split/source hash、fold、train/validation mapping hash 与 unavailable records 都进入 run manifest、每条 metrics 和 checkpoint provenance。

FSDP 用一个统一 root 覆盖 core diffusion、sidechain、refiner 和 viability，并使用 `DistributedSampler` 与每-rank 独立 prior/noise seed。Full model/optimizer checkpoint 由所有 rank collective 汇集、仅 rank 0 写盘；日志和 status 也只有 rank 0 写。非空 run directory 会 fail closed，禁止多 rank 争写或覆盖历史任务。

默认 prior 比例：

```yaml
open_chain: 0.40
single_crossing: 0.40
topology_corrupted: 0.20
```

训练目标分成三段：core flow + peptide/iso/clash/exactly-one；rotamer/χ 驱动的 Atom14 坐标；最大位移受限的 all-heavy refiner。coordinate-free viability head 同时使用正确 candidate 与 wrong-plug hard negative，checkpoint 必须严格加载全部四个模块。

## Construction

`k/p` 均为 zero-based。需要已训练 checkpoint：

```bash
python -m scripts.run_mini construction \
  --checkpoint runs/lassodiff-mini-dev/checkpoint-final.pt \
  --sequence LLQRNGRDRLILSKN --k 7 --p 9 \
  --samples 8 --steps 40 \
  --output candidate.pdb
```

Construction 默认 single-crossing procedural seed。所有样本仍必须经过 strict checker；若没有 strict-valid 样本，JSON 会保留逐项 rejection reason，不会把“生成了坐标”写成成功。

## Screening

```bash
python -m scripts.run_mini screening \
  --checkpoint runs/lassodiff-mini-dev/checkpoint-final.pt \
  --sequence LLQRNGRDRLILSKN --k 7 --p 9 \
  --samples 16 --steps 40
```

Screening 强制 open-chain prior，且输出：

```text
candidate_viability
unassisted_valid_rate
candidate_score = candidate_viability * unassisted_valid_rate
```

它不会使用 hard iso projection、topology projection 或强 topology guidance。第一版判别应解释为 `LASSO / UNSUPPORTED / ABSTAIN`，在真实 non-lasso 校准集完成前不要宣称绝对 `NON-LASSO`。

## Validation 与 release gate

现有多维 validation 指标继续保留：formed-amide distance/angles/plane、backbone bonds、exact crossing count、plug match、tail persistence、clash、RMSD/lDDT、sampler finite rate，以及按 candidate/length/acceptor 分层的统计。Mini 在 `lassodiff.validation.threading_mini` 独立实现 full-tail hard/soft 几何，聚合入口是 `lassodiff.validation.strict_lasso`；它不依赖工作区未提交的 V3 checker 改动。

Construction 与 screening 必须分别报告，所有记录至少携带 checkpoint、seed、candidate mapping、prior mode、sampler steps、projection/guidance flags 和 sample count。release 前仍需 locked validation、hard-negative calibration 与 paired comparison；训练 loss 下降不构成完成证据。

## 主要文件

```text
lassodiff/atom_schema_lasso.py       core-7、candidate、共价图
lassodiff/structure_processor.py     ASX/GLX 与 PDB 规范化
lassodiff/internal_coordinates.py    无模板 internal-coordinate builder
lassodiff/peptide_prior.py           open/single/corrupted priors
lassodiff/model_mini.py              core diffusion
lassodiff/sidechain_builder.py       Atom14、rotamer/chi
lassodiff/atom_refiner.py            bounded all-heavy refiner
lassodiff/candidate_viability.py     coordinate-free viability
lassodiff/validation/strict_lasso.py strict truth checker
lassodiff/data/mini_split.py         locked-test cluster 5-fold 合同
scripts/build_mini_cv_split.py       构建并审计 CV manifest
scripts/verify_mini_startup.py       训练前门禁
scripts/train_mini.py                三段训练
scripts/train_mini_cv.py             顺序执行 5 个四卡 folds
scripts/run_mini.py                  construction/screening
configs/lassodiff_mini.yaml          默认合同
tests/mini/                          mini correctness tests
```
