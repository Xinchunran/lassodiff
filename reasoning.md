# OpenDDE Reasoning 接入 LassoDiff：实施合同

文档版本：v1.0
日期：2026-08-01
状态：Implementation contract / test-first runbook
目标架构：`lassodiff_opendde_v3` / schema 3

本文是工程合同，不是概念建议。任何标记为 MUST 的要求必须由测试和 preflight verifier 强制执行。

## 1. 决策摘要

V3 使用 frozen OpenDDE residue reasoning，同时完成：

- sequence-level `LASSO / NON_LASSO / ABSTAIN` 判别；
- candidate-specific lasso 结构生成质量提升；
- 防止 silent fallback、假装加载权重、模块未进入 forward/gradient/optimizer/checkpoint 等形式实现。

```text
Sequence
  -> OpenDDE frozen residue reasoning -> s_res / z_res
       |-> sequence feasibility path (candidate-independent)
       |     -> H0/no-lasso + acceptor + plug + OOD/calibration
       `-> structure path (candidate-specific)
             -> reasoning adapter -> topology adapter(k_j,p_j)
             -> structural tokens -> geometry-aware diffusion
             -> candidate structures -> strict topology checker
```

### 1.1 不可妥协原则

1. Sequence 判别必须发生在 `k/p/ring/closure/plug` 注入之前。
2. 第一阶段 OpenDDE reasoning MUST frozen/eval；只训练 adapter、gate、hypothesis、topology、structural token 和 diffusion。
3. Sequence gate 不得通过 diffusion 结果定义标签；生成成功只能作为辅助结构证据。
4. 每个 candidate 使用自己的 `k_j/p_j/acceptor/atom_mask/target`。
5. OpenDDE 加载、hash、commit、schema 不匹配必须 fail closed；禁止 ESM-only/legacy fallback。
6. OpenDDE single/pair 必须实际影响输出；结构路径 MUST 读取 `z_res`。
7. 每个 diffusion block 必须从当前 `x_t` 重算动态几何。
8. Screening 禁止 hard projection 和强 topology guidance。
9. required modules 必须被调用、影响输出、按合同获得梯度、进入 optimizer 与 checkpoint manifest。

## 2. OpenDDE 复用边界

固定官方 commit `f607bb3c9ff299c0627ac20f5ef8e25d716ed46f`。审计确认 Pairformer 包含 triangle multiplication outgoing/incoming、triangle attention start/end、pair transition、single attention with pair bias 和 single transition。OpenDDE 当前为 preview，官方明确提示 CLI、JSON 与 checkpoint 可能变化，因此禁止追踪 floating `main`。

第一阶段复用 frozen residue single/pair reasoning；不直接迁移完整 OpenDDE atom diffusion，因为其 atom/token schema、atom-to-token mapping 与现有 7-atom lasso 表示不兼容。第一阶段结构路径保持 LassoDiff geometry-aware diffusion，便于归因 reasoning 的收益。

固定 fork 必须提供稳定 `forward_reasoning()`；禁止依赖匿名 module index 或不稳定字符串路径。输出 schema、commit 和 checkpoint SHA 必须进入 cache/checkpoint manifest。

## 3. 目标模块

```text
lassodiff/opendde_bridge/{checkpoint,feature_builder,reasoner,cache,schema}.py
lassodiff/{sequence_base_encoder,sequence_gate,hypothesis_head,ood_head}.py
lassodiff/{reasoning_adapter,topology_adapter,structural_tokens_lasso}.py
lassodiff/{geometry_features,score_head_lasso_v3,topology_checker,model_v3}.py
configs/lassodiff_opendde_v3.yaml
scripts/{cache_opendde_reasoning,train_sequence_gate,train_structure_v3,train_joint_v3,evaluate_v3}.py
```

## 4. 核心数据合同

```python
@dataclass(frozen=True)
class OpenDDEReasoningState:
    single: Tensor              # [B,L,Cs_open]
    pair: Tensor                # [B,L,L,Cz_open]
    token_mask: Tensor          # [B,L] bool
    residue_index: Tensor       # [B,L]
    sequence_hashes: tuple[str, ...]
    checkpoint_sha256: str
    opendde_commit: str
    feature_schema_version: int

@dataclass
class SequenceAssessment:
    no_lasso_logit: Tensor      # [B]
    acceptor_logits: Tensor     # [B,L]
    plug_logits: Tensor         # [B,L]
    candidate_logits: Tensor    # [B,M]
    sequence_embedding: Tensor  # [B,C]
    ood_score: Tensor           # [B]

@dataclass
class CandidateStructureTargets:
    coords: Tensor              # [B,M,L,A,3]
    atom_mask: Tensor           # [B,M,L,A]
    target_valid: Tensor        # [B,M]
```

MUST：state 不含 candidate `k/p`；padding single/pair 为零或全程 mask；provenance 不得为空；候选维在 candidate objective 前不得消失。

## 5. Strict bridge 与 cache

- checkpoint 必须存在且 SHA-256 等于配置；当前本机 `opendde.pt` hash 为 `7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc`。
- `load_state_dict(..., strict=True)`；missing/unexpected key 均 fatal。
- 若只载入 reasoning 子模块，使用显式 prefix whitelist，并由 manifest 固定 `expected_numel`；禁止 `loaded_numel > 0` 这类弱断言。
- frozen reasoner 在 `model.train()` 后仍保持 `eval()`，参数 `requires_grad=False` 且无 grad。
- cache key 至少包含 sequence SHA、OpenDDE commit、checkpoint SHA、feature schema、MSA/template flags 与 dtype。
- cache metadata mismatch 必须报错或显式重建；不得静默读取旧状态。

推荐 cache：`cache/opendde_reasoning/<cache_key>/{state.safetensors,metadata.json}`。缓存是 candidate-independent，不得按 LP_ID 单独重复相同 sequence。

## 6. Sequence gate 与无 topology leakage

`assess_sequence()` 的签名不得出现 `k/p/lasso_feats/is_ring/is_loop/is_tail/acceptor_index/candidate_rank/closure_edge`。Gate 输入仅为 frozen reasoning 经 trainable adapter 得到的 `s_res/z_res/token_mask`。

Gate MUST 同时使用：masked single global pooling、N-terminal single、N-terminal-to-all pair pooling、global long-range pair pooling。输出 H0/no-lasso、acceptor、plug、candidate hypothesis、embedding 与 OOD score。

非法 hypothesis 只能 mask，禁止 clamp 成合法索引。acceptor 必须为 ASP/GLU 且在 token mask；`p>k` 并满足配置的 loop gap。

三态规则：OOD 超阈值 -> `ABSTAIN`；`p_lasso` 低于 reject threshold -> `NON_LASSO`；高于 accept threshold -> `LASSO_PLAUSIBLE`；中间区域 -> `ABSTAIN`。阈值由 held-out negatives 的目标 FPR 标定，不以固定 0.5 替代。

## 7. Sequence 数据

Positive 需来自已知 lasso peptide；LassoPred candidates 只能作为 candidate weak labels，不能成为 sequence positive 的唯一证据。

Negative 最低包含：背景非-lasso peptide、composition-matched shuffle、acceptor/plug hard mutant、wrong-loop/topology 和 random/OOD。随机序列不得成为主要 negatives。

MUST 使用 topology-stratified sequence-neighbour split：等长序列用 normalized Hamming，变长序列用显式 indel-aware distance；同一 neighbour cluster、同 sequence 的所有 candidates、hard mutant 与来源 positive 不得跨 split。Topology strata 只能在完整 cluster 之间平衡，禁止为追求比例拆 cluster。Bootstrap 不用于构造 split。

## 8. Structure reasoning 路径

Reasoning adapter 分别对 OpenDDE single/pair 做 LayerNorm + projection。Topology adapter 在 shared reasoning 之后按 candidate 注入 ring/loop/tail membership、distance-to-k/p、closure、possible iso bond、plug/tail-to-ring relation 与 teacher prior。

Structural roles 至少包括 `BACKBONE/SIDECHAIN/NTERM_REACTIVE/ACCEPTOR_CARBOXYL/PLUG/RING_CONTEXT`，并显式输出 token single/pair、mask、parent residue、role id 与 atom map。

每个 diffusion block 从当前 `x_t` 重算 distance RBF、relative displacement/local orientation、NTERM-acceptor、plug-ring、tail-ring 与 role pair features。标量距离特征需 SE(3) invariant；向量特征 equivariant；最终坐标速度 equivariant。

Noise conditioning 使用 Fourier embedding of log noise，禁止继续把未规范化 raw sigma 直接送入普通两层 MLP。

## 9. Loss 与 gradient routing

```text
L_total = λ_gate L_gate + λ_hyp L_hypothesis
        + y_lasso λ_struct L_structure + λ_cal L_calibration + λ_ood L_ood
L_structure = -τ log Σ_j π_j exp(-loss_j/τ)
```

结构 prior 必须使用 `sequence_candidate_prior.detach()`，防止 diffusion 将 gate 拉向最易生成而非最可信 candidate。结构 loss 到 reasoning adapter 的初始 gradient scale 为 0.05--0.1。

- gate/hypothesis/calibration：更新 reasoning adapter 和对应 heads，不更新 topology/diffusion；
- structure：更新 scaled reasoning adapter、topology、structural tokens、diffusion，不更新 sequence/hypothesis heads；
- frozen OpenDDE：所有 loss 下均无 gradient。

## 10. 推理模式

Screening：先 gate；不运行 diffusion、不做 projection/guidance、不允许用户 `k/p` 改变 `p_lasso`。

Design：仅对 `LASSO_PLAUSIBLE` 或显式研发 override 运行结构生成。Override 输出必须同时标记原分类和 `generated_under_forced_topology=true`；强制生成结果不得反向修改 classification。

## 11. Test-first 阶段

1. P0 correctness：现有 loss/sampler/candidate/atom-mask 回归全绿。
2. P1 bridge/cache：missing checkpoint fatal、strict keys、loaded-numel manifest、frozen/eval、provenance、cache key/schema/roundtrip。
3. P2 gate：API AST isolation、single/pair output dependence、negative screening 不调用 diffusion、gate gradient isolation、OOD/ABSTAIN。
4. P3 structure reasoning：structure output 依赖 OpenDDE pair；candidate permutation；invalid mask 不 clamp；prior detached。
5. P4 structural tokens：roles/parent/pair/mask/output dependence。
6. P5 geometry diffusion：per-block call count、current-`x_t` dependence、SE(3) tests、behavioral ablation。
7. P6 joint/release：gradient routing、optimizer uniqueness/coverage、checkpoint manifest、sequence FPR、structure quality locked evaluation。

测试层级为 unit/contract/integration/regression/slow-release。AST/static scan 仅是辅助，必须有 runtime route、output dependence 和 behavioral tests。

## 12. Preflight 与 checkpoint

V3 preflight 必须检查：checkpoint/SHA/commit/schema、no fallback、reasoner frozen/eval、gate signature isolation、single/pair dependencies、required route/gradient/optimizer、screening no projection/guidance 与 manifest roundtrip。任一失败 exit 1，训练不得启动。

Checkpoint manifest 至少包含 architecture/schema、OpenDDE commit/checkpoint SHA、reasoning schema、reasoner frozen、required modules、allow_fallback=false、candidate-specific=true、screening projection=false。V1/V2 禁止 `strict=False` 进入 V3。

## 13. PR/实施顺序

P0 correctness -> bridge/cache -> sequence gate/negatives/screening -> reasoning structure adapter -> structural tokens -> dynamic geometry -> joint/release。一次变更不得同时跨越两个无法单独归因的核心架构阶段。

## 14. 完成定义

必须同时满足：gate 有显式 H0、无 topology leakage、依赖 OpenDDE single/pair、达到签署 FPR、OOD abstain；结构保持 candidate 独立、依赖 OpenDDE pair、roles 有效、geometry per block、topology/geometry release metrics 达标；工程上 strict load、no fallback、route/output/gradient/optimizer/checkpoint/ablation 证据完整。

详细指标与阈值见 [`metrics.md`](metrics.md)。

## 15. 当前实现与正式 medium 固定值（2026-08-01）

- OpenDDE real checkpoint preflight：strict 655,791,538 parameters，residue `single/pair` channels 均为 384。
- V3 trainable medium：`c_s=384, c_z=192, c_a=384, n_heads=8, diffusion_blocks=16`，共 29,948,119 parameters。
- FSDP：4 ranks、bf16 parameters、fp32 gradient reduction、每卡 batch 16、global batch 64。
- Structure phase 不把 656M frozen trunk 复制到各 rank；只读取 strict offline cache。Cache-only reasoner 不能 forward，cache miss 为 fatal。
- V3 topology/Hamming split v4：3,216 train / 410 validation / 403 test，3,122 unique sequences、1,453 single-linkage neighbour clusters；`topology_valid` structure records 为 2,674 / 339 / 353。
- 正式 cache 服务：`lassodiff-v3-opendde-cache-20260801.service`；每个 rank 独立 JSONL，成功完成后由 `scripts/verify_opendde_cache.py` 做全量 roundtrip manifest。
- 正式 trainer 必须同时读取 real preflight、full cache manifest、family split 和 architecture config；任何 provenance 或 count 不一致都在创建 run directory 前失败。

Sequence gate/joint 阶段的程序已构建，但当前仓库数据只能监督 structure。没有 verified positive/background negative manifest 时，`train_sequence_gate.py` 会因 evidence/negative strata 不完整而 fail closed；这不是 legacy fallback，也不能通过把 LassoPred weak candidates 改名为 positive 绕过。

### 15.1 Topology-loss 修复合同

20,000-step velocity-only baseline 的 strict topology pass 始终为零。审计确认旧 checker 错误要求第二个 carboxylate oxygen，而真实 formed isopeptide target 的该 atom mask 为 0%；旧 loader 还随机混用 open `minN` 与 closed `relaxN`。修复后 MUST：

- 形成的异肽酰胺使用 `NTERM-N/Ciso/一个 carbonyl O`；不得要求已被 N 替代的第二个 oxygen；
- structure target policy 为 `topology_valid`，优先同 candidate rank 的合格 `relaxN`，invalid target 显式 mask；
- 从 rectified-flow `x_t + (1-t)v_pred` 重建 endpoint，并同时优化 flow、backbone bond、iso distance、iso angle、iso plane proxy 与 target-relative differentiable signed crossing；checker 与 surrogate 必须共享 ring surface 和 plug-inclusive `p..tail` candidate thread，Gauss link 仅保留为诊断；
- topology auxiliary loss 保持 candidate dimension，进入 candidate marginal 之前不得 collapse；
- 2,000 step 必须出现非零 `iso_distance_valid` 与 `topology_pass`，否则保存 checkpoint 后 fail closed。
