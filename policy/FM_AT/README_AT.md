# FM_AT: Action-Time Flow Matching Policy

这个分支从 `policy/FM` 轻量复制而来，实现的是最终讨论的 **不使用 alpha、不使用显式门控** 的方案：

```text
Y = (A_L, A_R, t_L, t_R)
```

Flow Matching 直接学习从噪声到动作-时间联合 trace 的向量场：

```text
dY / d tau = v_theta(Y_tau, tau, O)
```

这和 `alpha * velocity` 的后处理不同：紧凑执行不是乘出来的，而是由生成变量里的时间通道表达。

## 核心文件

- `flow_matching_policy/policy/action_time_flow_unet_image_policy.py`
- `flow_matching_policy/config/robot_fm_at_16.yaml`
- `train_at.sh`

## 训练目标

```text
L = L_FM(Y)
  + lambda_energy * L_energy(Y)
  + lambda_world * L_world(z_t, Y_t -> z_{t+1})
  + lambda_mono * L_monotonic(t_L, t_R)
```

`L_FM` 使用 rectified-flow 形式：

```text
Y_tau = (1 - tau) * noise + tau * Y_data
target_velocity = Y_data - noise
```

## Compact trace 如何学习

原始 demo 没有紧凑轨迹时，zarr 不含 `time_trace`，dataset 默认给线性时间。真正的改进闭环是：

```text
Base FM_AT
-> sample action-time candidates
-> sim/WM rollout and score
-> select compact successful traces
-> save data/time_trace
-> distill back into FM_AT
```

推理时也可以直接多采样 rerank：

```text
score(Y|O) = E_trace(Y,O) + lambda_time * makespan(Y)
```

## 训练

```bash
cd policy/FM_AT
bash train_at.sh stack_bowls demo_clean 50 100 0
```

增加推理候选 rerank：

```bash
bash train_at.sh stack_bowls demo_clean 50 100 0 4
```

## 输出

`predict_action` 返回：

- `action`: 现有环境可执行动作。
- `action_pred`: 完整动作 chunk。
- `time_trace`: 生成的 `(t_L, t_R)`。
- `candidate_scores`: 多候选 energy + compactness 分数。
