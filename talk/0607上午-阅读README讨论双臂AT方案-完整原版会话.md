# 0607 上午：非对称双臂 AT 方案完整原版会话

导出时间：2026-06-07  
说明：本文件根据当前会话上下文可见内容整理，保留用户和助手的原始问答脉络；不包含系统/开发者提示和工具执行日志逐字内容。

---

## 1. 阅读 AT / WM README

### 用户

你先仔细阅读一下这几个readme 我们一起讨论一下实现方案  
以AT为主，WM是最早的版本，仅供参考使用

### 助手

我会先把这几个 README 和相关目录结构读一遍，重点抓 AT 的设计意图、FM_AT 与 DP_AT 的差异，以及 WM 里可借鉴但不该照搬的部分。读完我会先总结成几个实现讨论点，不急着改代码。

### 助手总结

我读完这几个 README，也顺手看了 AT 的核心 policy/config。我的理解是：现在真正要做的不是继续 WM 那条 `alpha/gate/warp` 路线，而是把 AT 做成一个干净的三段式：

**核心判断**

AT 的主干应该保持非常纯：

1. 主模型只生成 action chunk：`A = (A_L, A_R)`
2. coupling estimator 只估计左右臂依赖强度，不参与主动作混合
3. compact scheduler 在推理阶段搜索执行时序，不从 demo 学 `time_trace`

WM 里面有价值的部分只有两个：`energy rerank` 和 `latent world model risk`。`alpha`、`gate_sparsity`、`orthogonal_loss`、action warp 这些我建议不要带进 AT，否则研究边界会变脏。

**我看到的现状**

当前 AT 已经不是空壳，主要实现都在：

- `policy/FM_AT/flow_matching_policy/policy/coupling_scheduled_flow_unet_image_policy.py`
- `policy/DP_AT/diffusion_policy/policy/coupling_scheduled_diffusion_unet_image_policy.py`

里面已经有：

- `BimanualCouplingEstimator`
- `CompactScheduler`
- `ActionEnergyHead`
- `LatentActionWorldModel`
- `action_rerank_samples`
- `predict_action` 返回 `compact_schedule / makespan / coupling_scores / candidate_scores / wm_scores`

所以后续实现不是“搭框架”，更像是把方案打磨成真正可实验、可解释、可对比的版本。

**我建议的实现方案**

第一优先级是明确 `compact_schedule` 到底怎么被使用。现在 policy 返回了 schedule，但 `action` 仍然是普通同步 action chunk。如果 eval/deploy 环境不消费 `compact_schedule`，那它目前只能做候选 rerank 和日志分析，不能真正产生“紧凑执行”的物理收益。

所以我建议分两层做：

1. **Policy 层先稳定**
   - 保持 action-only DP/FM proposal。
   - coupling estimator 训练边缘/条件分支。
   - scheduler 只搜索 schedule，不改 action。
   - energy/WM 只作为 risk/rerank verifier。

2. **Execution 层再决定是否接 schedule**
   - 如果环境支持左右臂异步执行，就让 deploy/eval 使用 `compact_schedule`。
   - 如果环境只接受同步 16D action，那么先把 AT 作为“compactness-aware candidate selector”，不要声称已经完成真实异步压缩。

**我最想先讨论的两个问题**

1. 你希望我们先以 `DP_AT` 还是 `FM_AT` 为主实现？  
   我倾向先 `DP_AT`，因为 diffusion baseline 更标准，调试/对比更稳；`FM_AT` 可以随后同步同一套 scheduler/verifier 思路。

2. `compact_schedule` 后续是否会接入真实执行器？  
   这个决定 scheduler 返回值要不要从现在的 `(left_start, right_start)` 扩展成 `start + duration + arm_id/action_idx` 这种更完整结构。

我的建议路线：先把 `DP_AT` 做成 canonical AT，实现和日志打实；然后把同样的 AT 逻辑迁移/对齐到 `FM_AT`。WM 只作为 verifier/risk 的参考仓库，不再继承它的速度场和门控路线。

---

## 2. T3 耦合估计公式是否等价于原始想法

### 用户

这里面有个问题：

T3: 左右臂耦合估计  
辅助分支同时学习边缘和条件动作模型：

```text
p(a_L | O)
p(a_R | O)
p(a_L | a_R, O)
p(a_R | a_L, O)
```

用条件分支相对边缘分支的收益估计耦合：

```text
C_{L->R} = loss_marginal_R - loss_conditional_R
C_{R->L} = loss_marginal_L - loss_conditional_L
```

这些分数只用于调度约束和日志，不作为 gate，也不混合主 Flow 输出。

这个我记得我的想法不是直接loss_m - loss_cond，而是计算mar_distribution和cond_distribution的散度/熵，来判断两个分布之间的差距，你觉得他这个公式和我的想法是一致的吗

### 助手

不完全一致，但**有理论上的亲缘关系**。

如果这里的 `loss` 是严格的负对数似然 `-log p(a)`，并且 marginal/conditional 两个模型都拟合得足够好，那么：

```text
E[loss_marginal_R - loss_conditional_R]
= H(A_R | O) - H(A_R | A_L, O)
= I(A_R; A_L | O)
= E KL[p(A_R | A_L, O) || p(A_R | O)]
```

也就是说，在理想 NLL 情况下，“条件分支比边缘分支少多少 loss”确实等价于你说的**条件分布和边缘分布之间的差距 / 熵下降 / 条件互信息**。

但当前 README 里的公式写成：

```text
C_{L->R} = loss_marginal_R - loss_conditional_R
```

更像是一个**predictive gain proxy**，不是你原本说的显式 distribution divergence。尤其在我们这里，DP/FM 辅助分支通常训练的是 denoising MSE 或 velocity MSE，不是直接输出完整概率分布的 NLL。所以这个差值会受到模型容量、训练阶段、noise timestep、loss scale 的影响，不能严格解释成 KL/JS/entropy。

更关键的是，代码里推理阶段目前并不是直接用 `loss_m - loss_cond`，而是比较 conditional 和 marginal 分支输出的差异：

```text
(pred_cond - pred_marginal)^2
```

这个反而更接近“两个分布预测结果不同”的想法，但它仍然只是**输出均值/向量场差异**，不是显式分布散度。

所以我的判断是：

```text
README 公式：和你的想法方向一致，但表达得过于简化。
严格说，它只是 distribution divergence / entropy reduction 的 surrogate。
```

如果要更贴近你的原始想法，我建议 README 和实现里都改成这种表述：

```text
C_{L->R} = D[p(a_R | a_L, O) || p(a_R | O)]
C_{R->L} = D[p(a_L | a_R, O) || p(a_L | O)]
```

第一版实现可以用 denoiser/velocity 输出差异作为可计算 proxy：

```text
C_{L->R} ≈ || f_R^cond - f_R^marginal ||^2
C_{R->L} ≈ || f_L^cond - f_L^marginal ||^2
```

如果以后想更严谨，可以进一步做 ensemble variance、diffusion likelihood proxy、或者显式 Gaussian head 来估计 KL。我的建议是：论文/README 里用“distribution discrepancy / conditional information gain”，代码第一版用 MSE proxy，不要把它说死成 `loss_m - loss_cond`。

---

## 3. 接真实执行器、加速幅度、MSE proxy 与原始 KL 思路

### 用户

A1.你的Q2：compact_schedule 后续是否会接入真实执行器？  
我的回答：是的，这正是我想要的

A2.现在还有一个问题，加速多少，现在貌似通过cond和marginal来判别是否加速，那么加速多少，是如何实现的？

A3.你说现在用的方式是 `(pred_cond - pred_marginal)^2`，那我理解的是让marginal和cond的预测结果是一致的，那么这个不太符合事实，我觉得应该是两者的分布应该是趋于一致的，你觉得呢？如果这个地方的mse不是训练阶段的目标函数，那么就当我没说

A4. 你最后说的，目前的方法是：

```text
C_{L->R} ≈ || f_R^cond - f_R^marginal ||^2
C_{R->L} ≈ || f_L^cond - f_L^marginal ||^2
```

但是我的原始想法是：

```text
C_{L->R} = D[p(a_R | a_L, O) || p(a_R | O)]
C_{R->L} = D[p(a_L | a_R, O) || p(a_L | O)]
```

但是你又说，DP/FM 辅助分支通常训练的是 denoising MSE 或 velocity MSE，不是直接输出完整概率分布的 NLL

那么问题在与这两者哪个更好？

### 助手

是的，你的原始想法更准确。当前 README 那个 `loss_m - loss_cond` 只能算一个简化版 proxy，不应该当成最终定义。

A1 我先确认：既然 `compact_schedule` 会接真实执行器，那调度器后面必须返回更完整的信息，不只是现在的 `left_start/right_start`。至少要有：

```text
left_start, right_start
left_duration, right_duration
left_end, right_end
left_speed_scale, right_speed_scale
```

否则真实执行器不知道“每段动作要跑多久”。

A2：现在“加速多少”不是由 `cond/marginal` 直接决定的。

当前代码里的逻辑大概是：

```text
duration = base_dt * scale
scale ∈ [scheduler_min_duration_scale, scheduler_max_duration_scale]
```

现在 config 是：

```yaml
scheduler_min_duration_scale: 0.5
scheduler_max_duration_scale: 1.0
```

所以理论上每段最多压到原来 50% 时长，也就是局部最多约 `2x` 加速。

但 `cond/marginal` 只提供 coupling score，它回答的是：

```text
左右臂这一步是否强相关？
能不能错开 / 重叠 / 异步？
```

真正决定“加速多少”的是 scheduler objective：

```text
lambda_time * makespan
+ dependency_weight * coupling_constraint
+ dynamics_weight * action_change / duration
+ risk_weight * risk
```

直觉上：

```text
低耦合 + 动作变化小 + 风险低 -> duration 可以短，允许加速和重叠
高耦合 -> 更同步或保持 precedence
动作变化大 -> 不宜压太短
risk 高 -> 不选这个 candidate/schedule
```

不过现在实现还有一个明显缺口：scheduler 内部采样了 `duration`，但最后只返回了 `starts`，没有把 duration 暴露给执行器。所以如果要接真实执行器，这里必须改。

A3：你担心的点是对的，但当前代码里 `||pred_cond - pred_marginal||^2` 不是训练目标。

训练阶段是：

```text
marginal -> target
conditional -> target
```

没有训练：

```text
conditional == marginal
```

也就是说它没有强行让两者一致。推理时才用两者输出差异估计 coupling。

但你说“应该比较两个分布，而不是只比较预测结果”，这个我赞同。现在的 MSE 只比较 deterministic mean / denoising vector / flow vector。它会漏掉一种很重要的情况：

```text
marginal 和 conditional 的均值差不多，
但 conditional 分布明显更窄、更确定。
```

这种情况下其实存在耦合信息，但当前 `pred_cond - pred_marginal` 可能接近 0。

A4：两者哪个更好？

理论上，你的原始定义更好：

```text
C_{L->R} = D[p(a_R | a_L, O) || p(a_R | O)]
C_{R->L} = D[p(a_L | a_R, O) || p(a_L | O)]
```

它更符合“耦合 = 条件信息改变了动作分布”这个思想，也能表达 entropy reduction。

当前方法：

```text
C ≈ ||f_cond - f_marginal||^2
```

优点是简单、便宜、容易接 DP/FM；缺点是它只是 vector/score discrepancy proxy，不是真正的 distribution divergence。

我建议我们把方案定成三层：

```text
理论定义：
C = distribution divergence / conditional information gain

第一版实现：
用 denoiser/velocity 输出差异作为 proxy

更强实现：
让 marginal/conditional 分支输出 Gaussian mean + variance，
然后显式算 KL / symmetric KL / JS-like divergence
```

比如：

```text
p_marg = N(mu_m, sigma_m^2)
p_cond = N(mu_c, sigma_c^2)

C = KL[p_cond || p_marg]
```

这样就能同时比较：

```text
均值变化
方差变化
条件分布是否更确定
```

我的建议很明确：README 和论文表述应该改回你的原始想法，使用 `D[p_cond || p_marg]`。代码第一阶段可以保留 MSE proxy，但命名上要叫 `score_discrepancy_proxy` 或 `vector_field_discrepancy_proxy`，不要把它说成真正的 coupling divergence。后续如果我们要做得漂亮，就加 Gaussian/uncertainty head 来算 KL。

---

## 4. 计划：直接改成原始想法

### 用户

你直接帮我改成真正的我的原始想法吧

### 助手提出计划

把 `DP_AT` 和 `FM_AT` 的 T3 从 `loss_marginal - loss_conditional` / `pred_cond - pred_marginal` proxy，改成你的原始定义：

```text
C_{L->R} = KL[p(A_R | A_L, O) || p(A_R | O)]
C_{R->L} = KL[p(A_L | A_R, O) || p(A_L | O)]
```

辅助分支不再估计 denoising/velocity 输出差异，而是显式学习左右臂 clean normalized action chunk 的对角高斯分布。主 DP/FM action generator 保持 action-only，不加 gate、不混合输出、不 warp 动作。

核心修改包括：

- 在 `BimanualCouplingEstimator` 中替换现有 marginal/conditional denoiser 逻辑。
- 新建左右臂 action distribution heads。
- 每个 head 输出 `mu, logvar`，表示 clean arm action chunk 的 diagonal Gaussian。
- marginal 输入：`O`。
- conditional 输入：`O + other_arm_action_chunk`。
- 训练 loss 使用 Gaussian NLL，而不是 MSE。
- 推理时 `estimate()` 返回真正的 KL coupling。
- scheduler 增加真实执行器需要的信息：
  - `compact_schedule_durations`
  - `compact_schedule_ends`
  - `compact_schedule_speed_scale`

---

## 5. 不修改现有 DP_AT / FM_AT，复制成 AT2

### 用户

不直接在DP_AT和FM_AT的代码直接修改，而是拷贝这两个文件夹的代码，另行修改，比如可以叫DP_AT2, FM_AT2之类的，总而言之就是不直接在现有的代码上进行修改

### 助手提出新计划

新建两个实验分支：

```text
policy/DP_AT2
policy/FM_AT2
```

它们从当前 AT 分支完整复制而来，然后只在新目录里实现你的原始 coupling 定义：

```text
C_{L->R} = D[p(A_R | A_L, O) || p(A_R | O)]
C_{R->L} = D[p(A_L | A_R, O) || p(A_L | O)]
```

默认实现使用 diagonal Gaussian KL，而不是 `loss_marginal - loss_conditional` 或 `pred_cond - pred_marginal`。

计划要点：

- 复制完整目录，而不是只复制单个 policy 文件：
  - `policy/DP_AT -> policy/DP_AT2`
  - `policy/FM_AT -> policy/FM_AT2`
- 保留内部包名 `diffusion_policy` / `flow_matching_policy`。
- 只修改 `AT2` 分支：
  - README 改写为 distribution divergence / conditional information gain。
  - config 改名为 `robot_dp_at2_16.yaml`、`robot_fm_at2_16.yaml`。
  - `train_at.sh` 默认使用新 config，并把日志/exp name 改成 `*_at2_kl`。
  - eval/deploy 脚本里的 hardcoded policy name / checkpoint path 改为 `DP_AT2`、`FM_AT2`。
- 在 `AT2` 的 coupling estimator 中：
  - marginal head 学 `p(A_L | O)`、`p(A_R | O)`。
  - conditional head 学 `p(A_L | A_R, O)`、`p(A_R | A_L, O)`。
  - 每个分布输出 `mu, logvar`，表示 clean normalized action chunk 的对角高斯。
  - 训练 loss 使用 Gaussian NLL。
  - 推理 coupling score 使用 `KL[p_cond || p_marg]`。
- scheduler 返回真实执行器需要的信息：
  - `compact_schedule_durations`
  - `compact_schedule_ends`
  - `compact_schedule_speed_scale`

---

## 6. 实施 AT2

### 用户

PLEASE IMPLEMENT THIS PLAN:

新建 DP_AT2 / FM_AT2 并实现分布散度耦合估计。

要求：

- 不修改现有 `policy/DP_AT` 和 `policy/FM_AT`。
- 新建 `policy/DP_AT2` 和 `policy/FM_AT2`。
- AT2 使用 diagonal Gaussian KL。
- scheduler 返回 duration/end/speed scale。
- 训练和 eval/deploy 入口切到 AT2。
- 若目录已存在则停止确认，不覆盖已有实验目录。

### 助手实施记录

我开始实现了，会先确认 `DP_AT2/FM_AT2` 不存在，再复制目录；之后所有代码改动只落在新分支里。这里我会特别小心不碰现有 `DP_AT` 和 `FM_AT`，它们就留作 baseline。

确认结果：

- `policy/DP_AT2` 不存在。
- `policy/FM_AT2` 不存在。
- 工作树里只有未跟踪 `.venv/`，忽略。

执行：

- 复制 `policy/DP_AT -> policy/DP_AT2`
- 复制 `policy/FM_AT -> policy/FM_AT2`

然后在 AT2 中完成：

1. 修改 DP_AT2 policy：
   - 新增 `DiagonalGaussianActionHead`
   - `BimanualCouplingEstimator` 改成 action distribution estimator
   - `compute_aux_loss` 使用 Gaussian NLL
   - `estimate` 使用 `KL[p_cond || p_marg]`
   - scheduler 返回 `durations / ends / speed_scale`
   - `predict_action` 返回新增 schedule 字段
   - DP coupling estimator 使用 clean normalized actions，而不是 diffusion target

2. 修改 FM_AT2 policy：
   - 同步 DP_AT2 的 Gaussian KL estimator
   - FM coupling estimator 使用 clean normalized actions，而不是 flow velocity target
   - scheduler 和 `predict_action` 同步返回新增字段

3. 新增 AT2 config：
   - `policy/DP_AT2/diffusion_policy/config/robot_dp_at2_16.yaml`
   - `policy/FM_AT2/flow_matching_policy/config/robot_fm_at2_16.yaml`

4. 修改训练入口：
   - `policy/DP_AT2/train_at.sh` 默认 `robot_dp_at2_16`
   - `policy/FM_AT2/train_at.sh` 默认 `robot_fm_at2_16`
   - exp name 改为 `*_at2_kl`

5. 修改 eval/deploy：
   - `deploy_policy.yml` 改为 `DP_AT2` / `FM_AT2`
   - eval 脚本 policy name 改为 `DP_AT2` / `FM_AT2`
   - double env checkpoint path 改为 AT2

6. 修改 AT2 README：
   - README 主定义改为 KL / conditional information gain。
   - 说明 `compact_schedule_durations`、`compact_schedule_ends`、`compact_schedule_speed_scale`。

验证：

- `python3 -m py_compile` 覆盖两个 AT2 policy 文件和 train entry，通过。
- `python3 -m compileall -q policy/DP_AT2 policy/FM_AT2` 通过。
- `torch` smoke test 未运行，因为当前 `python3` 和 `.venv/bin/python` 都没有安装 `torch`。
- 清理了验证生成的 `__pycache__`。
- 最终 `git status` 显示只有新建 `policy/DP_AT2/` 和 `policy/FM_AT2/`，原 `DP_AT/FM_AT` 未改动。

### 助手最终回复

已实现，且没有修改原来的 `policy/DP_AT` / `policy/FM_AT`。新分支已创建：

- `policy/DP_AT2`
- `policy/FM_AT2`

主要完成了：

- AT2 README 改成你的原始定义：`KL[p_cond || p_marg]`
- 新增 config：`robot_dp_at2_16.yaml`、`robot_fm_at2_16.yaml`
- `train_at.sh` 默认切到 AT2 config 和 `*_at2_kl` 实验名
- eval/deploy 入口切到 `DP_AT2` / `FM_AT2`
- coupling estimator 改成 clean normalized action 的 diagonal Gaussian NLL 训练
- 推理 coupling score 改成逐步 KL：
  - `C_{L->R} = KL[p(A_R | A_L, O) || p(A_R | O)]`
  - `C_{R->L} = KL[p(A_L | A_R, O) || p(A_L | O)]`
- scheduler 现在额外返回：
  - `compact_schedule_durations`
  - `compact_schedule_ends`
  - `compact_schedule_speed_scale`

验证情况：

- 已跑 `py_compile` 和 `compileall`，通过。
- 原 `DP_AT/FM_AT` 未被改动，`git status` 只显示新增 `DP_AT2/FM_AT2`。
- 需要 `torch` 的 smoke test没法跑，因为当前 `python3` 和 `.venv/bin/python` 都没有安装 `torch`。

---

## 7. 逐步 KL 还是 action chunk KL

### 用户

推理 coupling score 改成逐步 KL：

```text
C_{L->R} = KL[p(A_R | A_L, O) || p(A_R | O)]
C_{R->L} = KL[p(A_L | A_R, O) || p(A_L | O)]
```

scheduler 现在额外返回：

```text
compact_schedule_durations
compact_schedule_ends
compact_schedule_speed_scale
```

你提到逐步的KL，我觉得对于action的执行，一般考虑一个action chunk的KL，是否比较好

### 助手

我同意你的直觉：**如果执行器是按 action chunk 作为一段轨迹来执行，那么 chunk-level KL 更自然**。

严格的定义应该是：

```text
C_{L->R} = KL[p(A_R^{1:H} | A_L^{1:H}, O) || p(A_R^{1:H} | O)]
C_{R->L} = KL[p(A_L^{1:H} | A_R^{1:H}, O) || p(A_L^{1:H} | O)]
```

也就是比较“整段右臂动作分布”有没有因为给定左臂整段动作而改变，而不是只看某一个 timestep。

不过逐步 KL 也不是错，它适合 scheduler 做局部决策：

```text
第 1-3 步低耦合，可以重叠/压缩
第 4-6 步高耦合，需要同步
第 7-10 步又低耦合，可以错开
```

所以我觉得更好的设计是两层：

```text
per-step KL: 给 scheduler 做局部 timing / duration / sync cost
chunk-level KL: 给 candidate rerank、整体耦合判断、日志和论文主指标
```

当前实现里的 diagonal Gaussian 其实已经接近这个结构。因为我们假设 action chunk 的分布在时间和 action dim 上是 factorized，所以：

```text
chunk_KL ≈ sum_t sum_d KL_t,d
```

或者为了数值尺度稳定：

```text
chunk_KL = mean_t KL_t
```

现在代码保留的是：

```text
coupling_scores: [B, T, 2]
```

我建议再加一个：

```text
coupling_chunk_scores: [B, 2]
```

例如：

```text
coupling_chunk_scores = coupling_scores.mean(dim=1)
```

如果担心短暂高耦合被 mean 稀释，也可以同时记录：

```text
coupling_chunk_mean
coupling_chunk_max
coupling_chunk_p90
```

我的推荐修改是：

1. `coupling_scores` 继续保留逐步 KL，供 scheduler 精细调度。
2. 新增 `coupling_chunk_scores`，作为主要理论指标和 action chunk 层面的耦合强度。
3. scheduler 可以先用 `mean + max` 混合：

```text
C_schedule = mean_t(KL_t) + beta * max_t(KL_t)
```

这样既不丢局部峰值，也不被单点噪声完全支配。

所以结论：**你的 action chunk KL 作为主定义更好；逐步 KL 适合作为 scheduler 的内部细粒度信号。** 我建议 AT2 里下一步就补 `coupling_chunk_scores`，README 也把主定义改成 chunk-level KL，per-step KL 作为 factorized/temporal decomposition。

---

## 8. 当前请求：导出本轮会话

### 用户

帮我把当前的会话的完整历史记录，总结到talk/文件夹下，给我，给我两个文件：

1. 完整原版会话内容
2. 整理总结后的会话内容

