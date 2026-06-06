# DP_WM: 非对称双臂世界模型 Flow Policy

这个目录是从 `policy/FM` 轻量复制出来的实验分支，不包含 checkpoints 和 zarr 数据。核心入口是：

- `flow_matching_policy/policy/asymmetric_world_model_flow_policy.py`
- `flow_matching_policy/config/robot_dp_wm_16.yaml`
- `train_wm.sh`

## 方法对应关系

把 markdown 里的 A/B/C 统一分解落成以下结构：

```text
v_L = v_L^loc + g_L * (v_L^cond - v_L^loc)
v_R = v_R^loc + g_R * (v_R^cond - v_R^loc)
```

其中：

- `v^loc` 对应弱耦合/局部方案 C 的 backbone。
- `v^cond - v^loc` 对应跨臂条件残差，近似 energy/log-prob residual。
- `g_L, g_R` 是可解释门控，配合 `gate_sparsity_loss` 鼓励默认走简单结构，需要时才打开残差。
- `orthogonal_loss` 约束左右 vector field 方向，减少左右臂动作场互相干扰。

## 世界模型

`LatentActionWorldModel` 实现 LeWorldModel-style 辅助目标：

```text
z_t = encoder(o_t)
z_hat_{t+1} = f(z_t, a_t)
L_world = ||z_hat_{t+1} - stopgrad(z_{t+1})||^2 + L_gaussian(z)
```

这不会替代动作 flow，而是给共享视觉 encoder 一个动态预测约束，让门控/残差更容易感知阶段、耦合和风险。

## 能量模型

`CouplingEnergyHead` 学一个标量能量：

```text
E(A_L, A_R, O)
```

训练时让真实左右动作配对能量低于 corrupted pairing：

```text
L_energy = softplus(E(pos) - E(neg) + margin)
```

推理时可设置：

```yaml
policy.energy_rerank_samples: 4
```

让 flow 采样多个 action chunk，再用 energy head 选择最低能量候选，相当于第一版 `flow proposal + EBM rerank`。

## 训练建议

第一阶段训练方向场、门控、world model 和 energy：

```bash
bash train_wm.sh stack_bowls demo_clean 50 100 0 base
```

第二阶段从 base checkpoint 只训练速度场：

```bash
bash train_wm.sh stack_bowls demo_clean 50 100 0 speed checkpoints/stack_bowls-demo_clean-50-100/600.ckpt
```

第三阶段小学习率 joint fine-tune 可用：

```bash
bash train_wm.sh stack_bowls demo_clean 50 100 0 joint checkpoints/stack_bowls-demo_clean-50-100-wm-speed/100.ckpt
```

## 主要日志

训练日志会额外记录：

- `world_model_loss`, `world_pred_loss`, `world_gaussian_loss`
- `energy_contrastive_loss`, `coupling_energy_positive`, `coupling_energy_negative`
- `gate_sparsity_loss`, `orthogonal_loss`, `residual_balance_loss`
- 继承自 factorized baseline 的 `factorized_w`, `factorized_u`



================

是的，`alpha` 这个加速/速度系数确实来自你的 markdown 讨论，不是我临时发明的。

最主要是在 [chatgpt-export_非对称双臂解耦方案.md](/home/lzy/code/BiCoord-Bench/policy/DP_RL/chatgpt-export_非对称双臂解耦方案.md:37) 里：

```text
dx/dτ = α(x,τ) v(x,τ)
```

后面还明确写了：

```text
αψ：局部速度，决定“走多快”
```

也有双臂版本：

```text
dz_L/dt = α_L(o,z) v_L(o,z)
dz_R/dt = α_R(o,z) v_R(o,z)
```

对应位置在 [chatgpt-export_非对称双臂解耦方案.md](/home/lzy/code/BiCoord-Bench/policy/DP_RL/chatgpt-export_非对称双臂解耦方案.md:620)。

另一个文件 [chatgpt-export_05_02 13_08 非对称双臂解耦方案对比.md](/home/lzy/code/BiCoord-Bench/policy/DP_RL/chatgpt-export_05_02%2013_08%20非对称双臂解耦方案对比.md:705) 里也有：

```text
vθ(x,τ,o) = αθ(x,τ,o) · uθ(x,τ,o)
```

这里 `αθ` 被解释成 task-aware 的速度场。

所以我刚才说的 `alpha` 主要继承的是你 markdown 里的 **speed field / time-reparameterization** 思路。区别只是我把它进一步放到了“紧凑轨迹生成”的问题里：`α_L, α_R` 不只是加速动作，还可以表达左右臂非对称执行频率。