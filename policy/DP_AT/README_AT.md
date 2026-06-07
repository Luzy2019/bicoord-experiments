# DP_AT: Coupling-Scheduled Action-Only Diffusion Policy

这个分支不再显式生成 `t_L, t_R`，也不使用 `alpha * velocity` 或显式门控。主模型只生成左右臂动作：

```text
A = (A_L, A_R)
```

时间不再是训练标签，而是外部 compact scheduler 的优化结果。整体方法是：

```text
Diffusion proposal
+ learned bimanual coupling estimator
+ compact schedule search
+ action energy / latent world model verifier
```

## 核心文件

- `diffusion_policy/policy/coupling_scheduled_diffusion_unet_image_policy.py`
- `diffusion_policy/config/robot_dp_at_16.yaml`
- `train_at.sh`

## 研究思想

Diffusion Policy 学习 action-only 分布：

```text
p(A_L, A_R | O)
```

紧凑执行不从 demo 的 `time_trace` 学，也不由模型直接输出。推理时采样多个动作候选，然后用耦合感知调度器搜索更短 schedule：

```text
minimize risk(A, O) + dependency(S, C) + dynamics(A, S) + lambda_time * makespan(S)
```

其中 `S` 是 scheduler 返回的执行时间表，`C` 是左右臂耦合强度。

## T3: 左右臂耦合估计

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

这些分数只用于调度约束和日志，不作为 gate，也不混合主 diffusion 输出。

## T1: 通用 compact scheduler

调度器使用非 task-specific 归纳偏置：

- 同臂动作顺序必须保持。
- 低耦合跨臂片段允许重叠。
- 高耦合跨臂片段增加 precedence / synchronization cost。
- 高动作变化片段增加 dynamics cost。
- WM uncertainty / action energy 高的候选增加 risk cost。

第一版使用 lightweight CEM-style schedule proposal search，只选择动作候选和 schedule，不 warp 动作轨迹。

## 训练

```bash
cd policy/DP_AT
bash train_at.sh stack_bowls demo_clean 50 42 0
```

增加推理动作候选 rerank：

```bash
bash train_at.sh stack_bowls demo_clean 50 42 0 4
```

## 输出

`predict_action` 返回：

- `action`: 现有环境可执行动作。
- `action_pred`: 完整动作 chunk。
- `compact_schedule`: scheduler 返回的 `(left_start, right_start)`。
- `makespan`: compact schedule 的总执行时长 proxy。
- `coupling_scores`: `(C_{L->R}, C_{R->L})` 的逐步估计。
- `candidate_scores`: 多候选综合分数。
- `wm_scores`: latent world model 风险分数。
- `collision_or_risk_cost`: energy / WM 风险代理代价。
