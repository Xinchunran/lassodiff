# LassoDiff V3 Topology Scoring Contract

文档版本：v1.0
日期：2026-08-01
状态：Core geometry aligned in V3.2；完整 checker truth-set/scoring release gate 尚未完成。

## 1. 目标与边界

下一轮结构工作的首要目标是：先建立可信、不可微的 authoritative threading checker，再用固定真值集校准可微 surrogate，最后把 checker 的组成项接入 candidate scoring。三者不得混为一个函数或用同一阈值循环自证。

```text
MD/template + synthetic controls
        -> authoritative checker（评估真值，不反传）
        -> signed threading label / crossing count / confidence
        -> differentiable surrogate（训练 loss）
        -> generated candidate
        -> authoritative checker + chemistry/geometry/clash metrics
        -> candidate score / hard rejection
```

Sequence gate 保持 candidate-independent。结构生成、threading checker 或 candidate score 均不得反向改写 `LASSO / NON_LASSO / ABSTAIN` 判别。

当前坐标 schema 是每个 residue 的 `N/CA/C/O` 加 acceptor 的 `Ciso/O1`，不是全原子表示。本文的 threading 以 covalent backbone/ring curve 为主；当前 clash 只能解释为 sparse-heavy-atom clash。

## 2. Authoritative threading checker

### 2.1 几何对象

每个 candidate 必须独立构造：

- ring curve：N-terminal 到 acceptor 的 backbone 路径，加 formed isopeptide closure；
- thread curve：acceptor/ring 后到 candidate plug `p_j` 及其后 tail 的 backbone 路径；
- candidate provenance：`record_id/rank/k_j/p_j/acceptor/target_name/atom_mask`。

不得用另一个 candidate 的 `p`、target 或 mask。缺失 closure atom、非有限坐标或非法 candidate 必须返回 `INVALID`，不能 clamp 后继续。

### 2.2 Checker 输出

Checker 至少返回以下独立字段，禁止只返回一个 `threading_success`：

```text
checker_valid
ring_closed
crossing_count
signed_crossing
threading_class        # {-1, 0, +1} 或经真值集签署的版本化类别
threading_confidence
plug_retained
tail_clearance
small_perturbation_stable
```

实现应使用非平面 ring 的稳定 spanning surface/triangulation，并对 segment-surface intersection 做方向和次数统计。Gauss-link 只能作为交叉验证信号；对开放 tail 必须显式声明 closure convention，不能把 `abs(link)>0.1` 当作未经校准的拓扑真值。

### 2.3 真值与稳健性门禁

固定、版本化 checker truth set：

- chemistry-qualified `relaxN` 正例，按 candidate rank 绑定；
- 从正例构造的 unthreaded、wrong-direction、wrong-plug、double-crossing decoys；
- 接近 ring plane、接近边界和轻微坐标噪声的困难样本；
- 刚体旋转/平移、residue padding 和 masked-atom 对照。

Release 前必须预注册并报告 checker confusion matrix。至少要求：刚体变换 bitwise/容差不变；小扰动不应无原因翻转类别；正负 decoy 可分；invalid 不计作 negative success。

## 3. Differentiable surrogate

V3.2 已修复核心错位：旧 checker 检查 pre-plug `k+1..p`，只有 219/6,433（3.40%）真实 target 恰好一次 crossing；旧 Gauss-link loss 又从 `p+1` 开始，跳过决定性的 `p→p+1` segment，其 hard-surface 对照只有 1,122/6,433（17.44%）。Plug-inclusive `p..tail` 在 6,211/6,433（96.55%）target 上恰好一次 crossing。因此训练与 checker 现统一使用 ordered CA ring 的 non-planar centroid-fan surface 和 candidate thread `p..tail`；Gauss link 仅作诊断。

Surrogate 的职责是提供梯度，不是定义真值。当前已落地 differentiable signed segment/surface crossing，并 target-relative 对齐；后续组成项包括：

- soft segment-to-ring-surface crossing；
- signed side-progression/crossing margin；
- 使用明确 closure convention 的 differentiable link term；
- plug retention 与 tail clearance；
- self-avoid barrier，阻止模型通过链相互穿越“作弊”改变 topology。

训练 target 使用 authoritative checker 产生的 signed class/crossing count，而不是只拟合未经校准的连续 Gauss-link 数值。所有 surrogate 分量必须单独记录 loss 和 gradient norm。

Surrogate 验收必须同时满足：

1. 在锁定 truth set 上与 checker label 单调相关，并报告 AUROC/AUPRC、rank correlation 和分层结果；
2. synthetic positive/negative 能过拟合；
3. 关闭该项会显著恶化 checker threading，而不是只让 surrogate loss 变大；
4. 2,000-step pilot 中 checker threading/topology 不再出现“train link loss 下降、rollout threading 同时下降”的持续背离。

## 4. Candidate scoring

Scoring 分成 hard validity 与 ranking 两层。

### 4.1 Hard rejection

以下任一失败，candidate 不得进入可交付结构集合：

- non-finite/sampler failure；
- formed-amide closure chemistry invalid；
- authoritative threading class 与 candidate target 不一致；
- severe sparse-atom clash 超过签署阈值；
- backbone bond/angle/plane 违反 hard limit。

### 4.2 Ranking score

只在 hard-valid candidates 之间排序。初始组成项：

```text
S = w_topology * checker_confidence
  + w_geometry * reactive_geometry_score
  + w_structure * normalized_structure_quality
  - w_clash * clash_penalty
  - w_instability * sampler_instability
```

权重只能在 validation 上校准并进入版本化 manifest；不得用 test 调权重。日志必须同时输出每个组成项、hard rejection reason 和总分，禁止只保留总分。Candidate score 可以用于 design-mode 排序，但不得成为 sequence positive 标签。

## 5. 物理几何与 clash

当前不是全原子模型，因此现有 `<1.2 Å` pair-rate 只能检测严重主链重叠。下一轮至少增加：

- any-clash structure rate、每结构 clash count、minimum nonbonded distance；
- 按原子角色与 sequence separation 分层的距离分布；
- backbone bond/angle/dihedral、peptide planarity、Ramachandran 分布；
- generated-vs-target distribution distance，而非仅均值；
- mean/median/p90/p95 和按 length/rank/acceptor/family 分层。

是否升级 atom14/atom37 全原子作为独立架构决策。在 sparse topology 与 checker 尚未稳定前，不把全原子升级和 surrogate 重构放入同一次实验。

## 6. 实施顺序

1. 冻结 candidate/rank 语义和 truth-set manifest；
2. 实现 authoritative checker v2 与独立单测；
3. 离线标注全部 chemistry-qualified targets/decoys；
4. 拟合并验证 differentiable surrogate；
5. 接入 structure objective，跑 synthetic overfit 和 2,000-step pilot；
6. 接入 hard rejection + ranking score；
7. 在相同 split/candidates/seeds/sampler steps 上与当前 checkpoint-1500/2000 配对比较；
8. 通过 behavioral ablation 后才启动更长训练。

任何阶段不得以 training loss 下降代替 checker threading、strict topology、clash 和 RMSD/lDDT 的联合证据。
