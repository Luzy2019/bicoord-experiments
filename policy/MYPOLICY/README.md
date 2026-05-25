我现在需要你做这几件事：

1.动作轨迹数据生成
是左右臂的任务，左臂和右臂分别是7dim的动作，例如：
left action:[a_l0, a_l1, a_l2, ..., a_l49, 0, 0, ...0], length=100
right action:[0, 0, ..., 0, a_r0, a_r1, a_r2, ..., a_r49], length=100
以上仅仅为例子，表明左右臂的动作轨迹是有关联或者无关联（上面的就属于无关联，你可以构造有关联的数据）

2.动作轨迹策略
输出双臂的动作轨迹，使用flow matching的方式

3.初始的结果
可能是双臂仍然保持数据集中的特性（先左臂执行再右臂执行），
但是我想训练的flow matchng方法，能让模型自动学会并行执行，例如：
训练数据：
[a_l0, a_l1, a_l2, ..., a_l49, 0, 0, ...0]
[0, 0, ..., 0, a_r0, a_r1, a_r2, ..., a_r49]
策略执行
[a_l0, a_l1, a_l2, ..., a_l49, 0, 0, ...0]
[a_r0, a_r1, a_r2, ..., a_r49, 0, 0, ...0]

请在给定的文件夹中生成对应的代码

## 实现思路

> **训练阶段绝不使用 `target`。** 训练数据只有 source（顺序示教轨迹）。
> `target` 字段仅在采样可视化时作为评测参考使用。

参考 `policy/FM` 下的 staged flow-matching + speed-modulation 实现，结合两份文档：

- **重参数化方法**：把"加速 / 并行化"理解为时间重参数化，
  *same path, different time parameterization*。
- **非对称双臂解耦**：把策略写成 *direction field × speed field*，
  双臂各自一套速度场 `α_L, α_R`，分阶段训练更稳定。

因此 MYPOLICY 采用两阶段：

```
Stage 1: 用 source 训练 base flow v_θ(x_t, t)（标准 rectified flow）
         → 学到顺序示教轨迹的任务流形 / 几何方向

Stage 2: 冻结 v_θ，单独训练「非对称 per-arm 重参数化头」
         (shift_L, scale_L, shift_R, scale_R)
         用 source 上的自监督目标驱动：
           * compactness  : 把每条臂的有效动作集中到 t=0 附近
           * content      : 不丢失有效动作内容（mass 守恒）
           * anchor       : shift 软对齐到 source 在线检测的 active-window 起点
           * fast         : 鼓励 scale > 1（时间压缩）
         → α_L, α_R 学到「先左后右 → 左右并行」的时间重参数化
```

推理时：从 base flow 采出一条类似 source 的顺序轨迹，再过 per-arm warp 头，
等价于 `α_L ≠ α_R` 的非对称时间重参数化，输出并行执行的双臂轨迹。

## 文件结构

- `trajectory_data.py`：合成数据生成
  - `source` 顺序示教（左先右后），训练唯一可见的数据
  - `target` 并行参考（只在评测/可视化时使用）
- `model.py`
  - `TrajectoryFlowMatchingPolicy`：Stage 1 unconditional rectified flow
  - `AsymmetricArmWarpHead`：Stage 2 per-arm `(shift, scale)` 预测头
  - `per_arm_affine_warp` / `asymmetric_warp_loss`：时间重参数化与自监督损失
  - `SpeedModulatedPolicy`：组合的推理策略
- `train.py`：`--stage 1` / `--stage 2`，全程只用 source
- `sample.py`：自动识别 checkpoint stage，输出 `prediction_raw`（裸 FM 输出）和 `prediction`（warp 后的并行结果）
- `visualize_data.py`：把 source / target / prediction_raw / prediction 画到 PNG

## 使用方法

在 `policy/MYPOLICY` 目录下执行：

```bash
# 0. 生成 source（target 字段仅做评测参考，不会进入训练）
bash generate_data.sh

# 1. Stage 1：训练 base flow（只用 source）
bash train.sh 1
# -> checkpoints/mypolicy_base.pt

# 2. Stage 2：冻结 v_θ，训练 per-arm 重参数化头（依然只用 source）
bash train.sh 2
# -> checkpoints/mypolicy_speed.pt

# 3. 采样
bash sample.sh stage1          # 看 base flow 默认输出（应为顺序轨迹）
bash sample.sh stage2          # 看加了 warp 后是否变并行
bash sample.sh stage2 warp     # 对真实 source 直接套 warp，验证 head 行为

# 4. 可视化
bash visualize_data.sh                                  # 数据集
bash visualize_data.sh outputs/mypolicy_samples_stage2.npz outputs/sample_vis_stage2
```

`bash train.sh` 与 `bash sample.sh` 的具体超参在脚本里调整。

## Stage 2 损失说明（与文档对应）

| 损失 | 默认权重 | 作用 | 对应文档概念 |
| --- | --- | --- | --- |
| `compactness` | 1.0 | 加权 `Σ (t/H) · ‖warp[t]‖ / Σ ‖warp[t]‖` 越小越好 | 把 active 区压向 t=0，即"FlowSpeedup 的低风险段加速" |
| `content` | 0.05 | `scale · Σ ‖warp‖ / Σ ‖source‖ ≈ 1`（相对误差） | 防止 head 把有效动作截断 / `trust region` |
| `anchor` | 2.0 | `|shift - online_active_start| / horizon` | 用 demo 自身在线检测的 active 起点提供方向先验 |
| `fast` | 0.0 | `-E[log scale]` | 文档中的 `-λ E[log α]` 加速奖励（默认关闭，避免压成"全压缩"的局部最优；想再加快可手动开启） |
| `scale_reg` | 1.0 | `(scale - 1)^2` | trust region：让 head 通过 *shift*（并行对齐）而不是 *scale*（一刀切压缩）来获得 compactness |

**关键约束**：所有损失都在 source 批次上即时计算，`detect_active_start` 操作的是 raw（未归一化）source；**没有任何离线 / 预计算的 target label**。Stage 2 训练期间 `source` 必须是 raw 空间，否则归一化会把 zero-padding 变成非零、破坏 active 检测（已在 `train.py` 里强制 raw）。

## 采样输出指标

`sample.py` 打印四个 `active_overlap`：

- `source`：原始顺序示教中左右臂同时运动的比例 → 接近 0
- `target (ref)`：并行参考 → 接近 1（仅供对照）
- `prediction_raw`：base flow 直接采样、未经 warp → 仍是顺序，接近 source
- `prediction`：经过 per-arm warp 头之后 → Stage 2 训练后应当显著高于 `source`，接近 `target`

在端到端冒烟测试中（512 样本，stage1=80 epochs，stage2=80 epochs），`mode=warp` 已可以达到 `prediction ≈ 0.85` overlap（对比 `source = 0.00`）。
**`generate` 模式**需要 base flow 充分训练（README 默认 2000 epochs）才能采出干净的"顺序"轨迹，否则 warp 效果会被噪声拖累。

## 与 `policy/FM` 的对应关系

| `policy/FM` | `policy/MYPOLICY` |
| --- | --- |
| `train_base_stage1.sh` | `bash train.sh 1` |
| `train_speed_stage2.sh` | `bash train.sh 2` |
| `SpeedModulationHead` （单一全局 α） | `AsymmetricArmWarpHead` （每臂独立的 shift+scale） |
| `speed_modulation_loss` | `asymmetric_warp_loss` （per-arm + compactness） |
| `warp_action_sequence` | `per_arm_affine_warp` |
