# LassoDiff Mini

`mini_dev` 是不依赖同源模板、OpenDDE 权重或全原子高斯初始化的小模型开发线。它直接替代旧的“template + classifier 选择 iso/plug”生成路径：

```text
sequence + candidate(k,p)
  -> frozen ESM residue conditioning
  -> random open-chain torsion prior
  -> dynamic-geometry torsion diffusion
  -> exact chain-kinematic backbone decoder
  -> residue-specific chi / rigid-group Atom14 reconstruction
  -> covalent-aware bounded all-heavy-atom refiner
  -> strict chemistry/topology checker
```

当前 active architecture：`lassodiff_mini_torsion_v2`，`schema_version = 2`。仓库尚未提供训练完成的新版 checkpoint，因此不能把程序化 seed、随机权重或 assisted seed 输出称为已训练生成结果。

## 开发边界

- Mini 的 backbone/reactive core schema 保持 `N/CA/C/O/CB/CISO/OISO`；生产状态是 `phi/psi/omega + acceptor chi`，不是 Cartesian core diffusion。
- `ASP_ISO` 使用 `CISO=CG, OISO=OD1`；`GLU_ISO` 使用 `CISO=CD, OISO=OE1`。
- formed acceptor 只有一个 carbonyl oxygen；不存在的 atom 必须由 mask 表示，禁止用零坐标冒充。
- V2 按 `(record_id, sequence, k, p)` 聚合 conformers，缺失 conformer 由 mask 表示，不复制 rank；非法 candidate 直接失败，不做 `clamp()`。
- ESM residue encoder 必须在 active forward graph 中执行、冻结、eval、detach；cache 不得包含 target、candidate label 或 fold 信息。
- headline `unassisted` 只允许 open-chain prior；single-crossing 只能作为显式 `assisted` secondary mode，不能合并 success rate。
- 旧 Cartesian core-7 与 frame-v1 checkpoint 是 legacy，不能以 V2 architecture ID 加载或静默 warm-start。
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

preflight 会行为验证：ESM active route、ASX 单氧 chemistry、无模板/open-chain prior、0/1/2 crossing fixtures、current-coordinate dependence、dynamic geometry、sidechain/covalent graph、screening 无 projection/guidance 和 strict checker route。任一失败都不得训练。

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

## Legacy 训练路径与 active V2 gate

下面原有 `scripts.train_mini`、`scripts.train_mini_cv`、`scripts.run_mini` 和
`configs/lassodiff_mini.yaml` 命令只用于复现旧 core-7/frame 实验，不是
`mini_dev` 新版生产入口。它们不得加载为
`lassodiff_mini_torsion_v2`，也不得把 single-crossing construction 结果
计入新版 unassisted 指标。

新版入口文件为：

```text
scripts/cache_mini_esm.py
scripts/train_mini_v2.py
scripts/overfit_mini_v2.py
scripts/evaluate_mini_v2.py
configs/lassodiff_mini_v2.yaml
```

新版正式训练前必须完成：

```bash
python -m pytest tests/mini_v2 -q -m "not slow"
python -m pytest tests/mini_v2 -q
```

任何 preflight、overfit 或 strict evaluation 失败都必须 fail closed，不能
继续 full training 或五折 rollout。

### Legacy commands

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

短周期改进实验可从完整 checkpoint 恢复并显式选择 profile。`combined` 使用 strict checker 生成的质量 manifest，但 validation 仍是完整 held-out fold：

```bash
python scripts/build_mini_quality_manifest.py \
  --metadata /path/to/lassopred.data.json --structure-root /path/to/structure \
  --split data/mini_cv5_locked_test_v1.json --fold 0 \
  --output artifacts/mini/quality_fold0_v1.json

python scripts/train_mini.py ... --profile combined \
  --quality-manifest artifacts/mini/quality_fold0_v1.json \
  --resume-checkpoint /path/to/checkpoint.pt --steps 8500
```

profiles 为 `baseline`、`screening`、`structure`、`combined`。质量课程只改变训练采样，不加载 locked test。

四卡 fail-fast 实验由一个顺序 launcher 管理：A/B/C/D 各先从 step 8000 运行 250 steps，随后四卡各评估一个 checkpoint；安全门和 construction/screening 等权早期信号最多选择两个改进分支，再与 baseline 各运行 250 steps。任何既无早期收益、又有 formed/backbone/clash、RMSD/lDDT 或 sampler 稳定性回退的分支不会继续。

```bash
python scripts/run_mini_fast_ablation.py \
  --resume-checkpoint runs/.../checkpoint-00008000.pt \
  --output-root runs/mini-fast-ablation-01
```

launcher 不调用 `scancel`、不覆盖非空目录，每个训练分支均通过 `torchrun --nproc-per-node=4` 使用同一组四卡。阶段决策分别写入 `stage1-selection.json` 与 `final-selection.json`。

frame-v1 是独立 opt-in geometry path，旧 Cartesian checkpoint 只能通过显式 warm-start 加载共享 trunk；旧 score head 和 optimizer 不会加载。先校准当前 fold 的 train-only template，再用 500-step early gate：

```bash
python scripts/calibrate_backbone_template.py \
  --metadata data/lassopred.data.json --structure-root Protenix/fine-tuning/structure \
  --cv-split data/mini_cv5_locked_test_v1.json --fold 0 \
  --output artifacts/mini/frame_template_fold0_v1.json

python scripts/train_mini.py ... --geometry frame_v1 \
  --frame-template artifacts/mini/frame_template_fold0_v1.json \
  --frame-warm-start-checkpoint runs/.../checkpoint-00008000.pt \
  --steps 500 --save-every 500 --validate-every 500

python scripts/check_frame_early_stop.py \
  --metrics runs/.../metrics.jsonl --step 500 --output runs/.../frame-early-decision.json
```

frame-v1 保持当前 Mini core-7 顺序 `N,CA,C,O,CB,CISO,OISO`，并使用独立 `sample_rectified_frame_flow`；legacy `sample_mini` 和最终 evaluator 不变。完整设计与保护边界见 `patch_frame.md`。

FSDP 用一个统一 root 覆盖 core diffusion、sidechain、refiner 和 viability，并使用 `DistributedSampler` 与每-rank 独立 prior/noise seed。Full model/optimizer checkpoint 由所有 rank collective 汇集、仅 rank 0 写盘；日志和 status 也只有 rank 0 写。非空 run directory 会 fail closed，禁止多 rank 争写或覆盖历史任务。

旧 Cartesian mini 的默认 prior 比例仅用于 legacy reproduction；新版生产训练 prior 为：

```yaml
open_chain: 0.80
broad_ramachandran: 0.20
single_crossing: 0.00
```

训练目标分成三段：core flow + peptide/iso/clash/exactly-one；rotamer/χ 驱动的 Atom14 坐标；最大位移受限的 all-heavy refiner。coordinate-free viability head 同时使用正确 candidate 与 wrong-plug hard negative，checkpoint 必须严格加载全部四个模块。

## Legacy Construction

`k/p` 均为 zero-based。需要已训练 checkpoint：

```bash
python -m scripts.run_mini construction \
  --checkpoint runs/lassodiff-mini-dev/checkpoint-final.pt \
  --sequence LLQRNGRDRLILSKN --k 7 --p 9 \
  --samples 8 --steps 40 \
  --output candidate.pdb
```

新版 headline rollout 使用 random open-chain torsion prior。所有样本必须经过现有 `lassodiff.validation.strict_lasso`；若没有 strict-valid 样本，JSON 会保留逐项 rejection reason，不会把“生成了坐标”写成成功。single-crossing 只能在 `assisted` 报告中单独出现。

## Legacy Screening

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

新版 paired rollout 使用 `scripts/evaluate_mini_v2.py` 及 `lassodiff.evaluation_mini_v2` 合同，必须同时报告 `prior_only_unassisted`、`untrained_model_unassisted`、`trained_model_unassisted`、`trained_model_assisted`。固定 sequence、candidate、seed set、sample count 和 sampler steps；每个 sample 的最终有效性只能来自 strict checker。synthetic negatives 不能替代真实 non-lasso 校准。

## 主要文件

```text
lassodiff/atom_schema_lasso.py       core-7、candidate、共价图
lassodiff/structure_processor.py     ASX/GLX 与 PDB 规范化
lassodiff/internal_coordinates.py    无模板 internal-coordinate builder
lassodiff/peptide_prior.py           open/single/corrupted priors
lassodiff/model_mini.py              core diffusion
lassodiff/model_mini_v2.py           active residue-level torsion diffusion
lassodiff/data/mini_grouped_pdb_dataset.py  grouped rank-matched targets
lassodiff/batch_mini_v2.py            real circular-flow batch builder
lassodiff/lasso_core_decoder.py       backbone + acceptor-chi decoder
lassodiff/metrics_mini_v2.py          rollout/RMSD/lDDT metrics
lassodiff/sidechain_builder.py       Atom14、rotamer/chi
lassodiff/atom_refiner.py            bounded all-heavy refiner
lassodiff/candidate_viability.py     coordinate-free viability
lassodiff/validation/strict_lasso.py strict truth checker
lassodiff/data/mini_split.py         locked-test cluster 5-fold 合同
scripts/build_mini_cv_split.py       构建并审计 CV manifest
scripts/verify_mini_startup.py       训练前门禁
scripts/verify_mini_v2_startup.py    V2-only behavioral preflight
scripts/train_mini.py                三段训练
scripts/train_mini_v2.py             V2 FSDP training route
scripts/run_mini_v2.py               V2 open-chain/assisted rollout
scripts/train_mini_cv.py             顺序执行 5 个四卡 folds
scripts/run_mini.py                  construction/screening
configs/lassodiff_mini.yaml          默认合同
configs/lassodiff_mini_v2.yaml       active torsion mini_dev 合同
tests/mini/                          legacy mini correctness tests
tests/mini_v2/                       active torsion/chemistry/strict contract tests
```

## Active mini_dev release gates

### V2 full training route

`lassodiff_mini_torsion_v2` 使用真实 grouped-PDB 路由：每个
`(record_id, sequence, k, p)` 是一个样本，最多三个 rank conformer 保留在
conformer 维度。批次从 open-chain torsion prior 与目标 conformer 构造
circular flow，经 frozen ESM/cache conditioner、动态几何 torsion 网络、
acceptor-chi lasso-core decoder 和多 conformer endpoint loss 完成反向传播。
推理路径不读取 target/template coordinates。

V2 必须使用独立 preflight；legacy `verify_mini_startup.py` 的 PASS 不能
授权 V2：

```bash
python -m scripts.verify_mini_v2_startup \
  --metadata data/lassopred.data.json \
  --structure-root structure \
  --source-split data/lassopred.lmdb/split_topology_hamming_v4.json \
  --cv-split data/mini_cv5_locked_test_v1.json \
  --output artifacts/mini_v2/preflight.json
```

fast gate、真实单候选 overfit gate 和 fold-0 pilot 通过前不得启动五折生产
训练。最终 rollout 必须回到同一个 `lassodiff.validation.strict_lasso`，并
分开报告 open-chain unassisted 与 single-crossing assisted 结果。

新版 mini_dev 在任何 full training 或五折 rollout 前必须通过：

```bash
python -m pytest tests/mini_v2 -q -m "not slow"
python -m pytest tests/mini_v2 -q
```

slow gate、locked validation 和最终 Lasso 评价都必须调用同一个 `strict_lasso_check`。teacher-forced loss、坐标有限性、backbone bond validity、single-crossing seed 成功率和 checkpoint keys 都不能单独构成 release 证据。
