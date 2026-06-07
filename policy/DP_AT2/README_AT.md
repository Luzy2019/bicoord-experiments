# DP_AT2: KL-Coupled Action-Only Diffusion Policy

这个分支从 `policy/DP_AT` 复制而来，用作新的 AT2 实验分支，不修改原始 `DP_AT` baseline。

主模型仍然只生成左右臂动作：

```text
A = (A_L, A_R)
```

时间不作为训练标签，也不由 diffusion 模型直接输出。紧凑执行由外部 scheduler 产生，并可交给真实执行器消费。

```text
Diffusion proposal
+ learned bimanual distribution-divergence coupling estimator
+ compact schedule search
+ action energy / latent world model verifier
```

## 核心文件

- `diffusion_policy/policy/coupling_scheduled_diffusion_unet_image_policy.py`
- `diffusion_policy/config/robot_dp_at2_16.yaml`
- `train_at.sh`

## T3: 分布散度耦合估计

辅助分支显式学习 clean normalized action chunk 的对角高斯分布：

```text
p(A_L | O)
p(A_R | O)
p(A_L | A_R, O)
p(A_R | A_L, O)
```

耦合强度定义为 conditional distribution 相对 marginal distribution 的 KL：

```text
C_{L->R} = KL[p(A_R | A_L, O) || p(A_R | O)]
C_{R->L} = KL[p(A_L | A_R, O) || p(A_L | O)]
```

这更接近 conditional information gain / entropy reduction：如果另一只手的动作显著改变了当前手的动作分布，耦合分数就高。实现中每个分布输出 `mu, logvar`，训练使用 Gaussian NLL；推理时按 action dim 求均值得到逐步 `coupling_scores`。

这些分数只用于调度约束和日志，不作为 gate，也不混合主 diffusion 输出。

## T1: Compact Scheduler

调度器搜索动作候选和执行时间表：

```text
minimize risk(A, O) + dependency(S, C) + dynamics(A, S) + lambda_time * makespan(S)
```

其中：

- 同臂动作顺序保持。
- 低耦合跨臂片段允许重叠和更短 duration。
- 高耦合跨臂片段增加 precedence / synchronization cost。
- 高动作变化片段增加 dynamics cost，避免过度压缩。
- WM uncertainty / action energy 高的候选增加 risk cost。

调度器返回 start、duration、end 和 speed scale。加速比例定义为：

```text
speed_scale = base_dt / duration
```

## 训练

```bash
cd policy/DP_AT2
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
- `compact_schedule_durations`: 每步左右臂 duration。
- `compact_schedule_ends`: 每步左右臂 end time。
- `compact_schedule_speed_scale`: `base_dt / duration` 的加速比例。
- `makespan`: compact schedule 的总执行时长 proxy。
- `coupling_scores`: `(C_{L->R}, C_{R->L})` 的逐步 KL 估计。
- `candidate_scores`: 多候选综合分数。
- `wm_scores`: latent world model 风险分数。
- `energy_scores`: action energy verifier 分数。
- `collision_or_risk_cost`: energy / WM 风险代理代价。
