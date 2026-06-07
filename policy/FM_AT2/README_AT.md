# FM_AT2: KL-Coupled Action-Only Flow Matching Policy

这个分支从 `policy/FM_AT` 复制而来，用作新的 AT2 实验分支，不修改原始 `FM_AT` baseline。

主模型仍然只生成左右臂动作：

```text
A = (A_L, A_R)
```

时间不作为训练标签，也不由 Flow Matching 模型直接输出。紧凑执行由外部 scheduler 产生，并可交给真实执行器消费。

```text
Flow proposal
+ learned bimanual distribution-divergence coupling estimator
+ compact schedule search
+ action energy / latent world model verifier
```

## 核心文件

- `flow_matching_policy/policy/coupling_scheduled_flow_unet_image_policy.py`
- `flow_matching_policy/config/robot_fm_at2_16.yaml`
- `train_at.sh`

## 研究思想

Flow Matching 学习 action-only 向量场：

```text
dA / d tau = v_theta(A_tau, tau, O)
```

训练时使用 rectified-flow：

```text
A_tau = (1 - tau) * noise + tau * A_data
target_velocity = A_data - noise
```

耦合估计不再用 velocity branch 差异作为定义，而是单独学习 clean normalized action distribution。

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

这些分数只用于调度约束和日志，不作为 gate，也不混合主 Flow 输出。

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

### DAG Compact Scheduler

AT2 默认使用 `scheduler_mode: dag`。调度器把每个动作步拆成图节点：

```text
L_i, R_i
```

固定同臂顺序边为 `L_i -> L_{i+1}`、`R_i -> R_{i+1}`。跨臂边由 coupling KL 产生：

- `C_{L->R}` 明显强于 `C_{R->L}` 时，建立 `L_i -> R_i`。
- `C_{R->L}` 明显强于 `C_{L->R}` 时，建立 `R_i -> L_i`。
- 双向都强且差距小，不建立双向环，改为 sync 约束。

DAG cost 包含 precedence violation、sync violation、critical path/slack 加权项，并继续保留 makespan、dynamics 和 risk 代价。

## Preference Energy

旧的 `ActionEnergyHead` 只表示 demo-likeness。新版本使用 `PreferenceEnergyHead`，对 action + schedule 输出分解能量：

```text
demo_energy
compactness_energy
dag_energy
dynamics_energy
phase_energy
total_energy
```

训练时构造三类样本：

```text
improved: expert action + DAG compact schedule
expert:   expert action + baseline schedule
bad:      shuffled action / over-compressed schedule / DAG-violating schedule
```

排序目标：

```text
E(improved) < E(expert) < E(bad)
```

因此 energy 不再只表示“像训练集”，而是用于表达更快、更符合 DAG、更低动态风险、更符合阶段奖励的偏好。

## Two-Stage BC/RL

训练阶段由 `training.stage` 控制：

```text
bc    : 离线 Flow Matching / BC，学习稳定成功策略
rl    : 从 BC checkpoint 初始化，在线 rollout，自模仿 RL 压缩时间
joint : BC batch 和 RL batch 混合训练
```

示例：

```bash
bash train_at.sh stack_bowls demo_clean 50 100 0 4 bc
bash train_at.sh stack_bowls demo_clean 50 100 0 4 rl checkpoints/xxx/600.ckpt
```

RL 阶段使用 reward-weighted flow matching 做 self-imitation，并用 PPO-like clipped objective 更新 candidate reranker / PreferenceEnergyHead。Flow Matching 主模型不直接使用标准 PPO log-prob。

默认 reward：

```text
success bonus
+ stage_eval_score 正增量
+ 越早完成阶段越高的 early phase reward
- env step / makespan
- DAG violation
- world model dynamics risk
- speed scale 超限
- collision / final failure
```

IsaacLab 环境建议在 `info` 中提供：

```text
stage_eval_score
success
collision
phase_id
task_time
```

如果 IsaacLab 环境支持 compact schedule execution，runner 会传入 action chunk、compact schedule 和 durations；否则 schedule 只用于 rerank 与 reward proxy。

## 训练

```bash
cd policy/FM_AT2
bash train_at.sh stack_bowls demo_clean 50 100 0
```

增加推理动作候选 rerank：

```bash
bash train_at.sh stack_bowls demo_clean 50 100 0 4
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
- `energy_scores`: PreferenceEnergyHead 的 `total_energy`。
- `collision_or_risk_cost`: energy / WM 风险代理代价。
- `dag_precedence_cost`, `dag_sync_cost`, `dag_critical_cost`: DAG 调度诊断。
- `preference_energy_components`: 分解 preference energy。
