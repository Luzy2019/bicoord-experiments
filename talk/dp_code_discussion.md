# RoboTwin DP 代码讨论记录

本文整理了围绕 `policy/DP` 目录中 Diffusion Policy 实现的代码阅读和解释。

## `dp_model.py` 中 `get_policy` 的作用

`policy/DP/dp_model.py` 中的 `get_policy()` 用来从训练好的 checkpoint 恢复 policy 模型，并切换到推理模式。

流程如下：

1. 使用 `torch.load()` 加载 checkpoint。
2. 从 checkpoint 中取出 Hydra 配置 `cfg`。
3. 根据 `cfg._target_` 动态实例化 workspace。
4. 调用 `workspace.load_payload()` 恢复模型权重和训练状态。
5. 默认使用 `workspace.model`。
6. 如果训练时启用了 EMA，则使用 `workspace.ema_model`。
7. 将模型移动到指定设备，例如 `cuda:0`。
8. 调用 `policy.eval()` 切换到推理模式。

简而言之：

```text
checkpoint -> cfg -> workspace -> model/ema_model -> device -> eval -> policy
```

## Workspace 是什么

`workspace` 不是单纯的 trainer，但它承担了训练管理器和实验容器的角色。

以 `RobotWorkspace` 为例，它负责：

- 创建 policy 模型
- 创建 EMA 模型
- 创建 optimizer
- 保存训练状态，例如 `global_step` 和 `epoch`
- 执行训练流程 `run()`
- 保存和加载 checkpoint

所以如果把 trainer 理解成“负责训练循环的对象”，`RobotWorkspace` 很像 trainer。但更准确地说，它是一个 workspace，范围比 trainer 更宽。

在部署时，`dp_model.py` 借助 workspace 只是为了按照训练时配置重建模型并加载权重，不是为了继续训练。

## 当前 DP 使用的是 UNet Diffusion

当前 `policy/DP` 配置使用的是 UNet 版 Diffusion Policy，不是 Transformer 版 diffusion。

配置文件中：

```yaml
policy:
  _target_: diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy
```

而 `DiffusionUnetImagePolicy` 内部创建的是：

```python
model = ConditionalUnet1D(...)
```

因此当前路径是：

```text
图像观测 -> ResNet encoder -> obs feature
obs feature + noisy action trajectory -> ConditionalUnet1D
DDPM scheduler -> denoised action trajectory
```

仓库中确实有 `TransformerForDiffusion`，但当前 `robot_dp_14.yaml` / `robot_dp_16.yaml` 没有把它接入主 policy。

## Transformer Diffusion 和 UNet Diffusion 的区别

两者都可以作为 diffusion 中的去噪网络，区别是网络骨干不同。

UNet diffusion：

- 当前 RoboTwin DP 实际使用
- 主干是 `ConditionalUnet1D`
- 使用 1D 卷积、downsample、upsample、skip connection
- 适合连续动作轨迹
- 训练通常更稳定、成本更低

Transformer diffusion：

- 实现在 `model/diffusion/transformer_for_diffusion.py`
- 把 action 序列的每个时间步看作 token
- 使用 self-attention 建模时间步之间的关系
- 可支持 encoder-only 或 encoder-decoder
- 对长序列和复杂依赖更灵活，但通常更吃数据和算力

结论：

```text
Diffusion 是生成框架；
UNet / Transformer 是其中用来预测噪声的网络骨干。
```

## `checkpoint_util.py` 的作用

`checkpoint_util.py` 中的 `TopKCheckpointManager` 用来管理只保留 Top-K 个最好的 checkpoint。

它本身不负责真正保存模型，而是负责判断：

- 当前 checkpoint 是否值得保存
- 如果值得保存，路径是什么
- 如果已经超过 K 个，要删除哪个旧 checkpoint

例如设置：

```python
TopKCheckpointManager(
    save_dir="checkpoints",
    monitor_key="val_loss",
    mode="min",
    k=3,
)
```

表示最多保留 `val_loss` 最小的 3 个 checkpoint。

## `cv2_util.py` 的作用

`cv2_util.py` 是图像可视化和图像处理工具。

主要函数：

- `draw_reticle()`：在图像上画准星。
- `draw_text()`：在图像上画带描边的多行文字。
- `get_image_transform()`：按目标分辨率 resize、center crop，并可做 BGR/RGB 通道转换。
- `optimal_row_cols()`：根据相机数量和最大分辨率，计算多相机画面拼接布局。

我们在 `policy/DP/diffusion_policy/common/__test__/cv2_util_test.py` 中添加了一个简单测试，并增加了可视化输出逻辑，用真实图片生成处理前后的 demo 图。

## `normalize_util.py` 的作用

`normalize_util.py` 用来根据数据统计量创建归一化器。

最常见的是把数据映射到 `[-1, 1]`：

```text
normalized = raw * scale + offset
```

主要函数：

- `get_range_normalizer_from_stat()`：按 `min/max` 归一化到 `[-1, 1]`。
- `get_image_range_normalizer()`：图像 `[0, 1]` 到 `[-1, 1]`。
- `get_identity_normalizer_from_stat()`：不做归一化。
- `array_to_stats()`：从 numpy 数组计算 `min/max/mean/std`。
- `robomimic_abs_action_*_normalizer_from_stat()`：针对机器人 action 的位置、旋转、夹爪等维度做特殊归一化。

它保证训练和推理时 action / image 的数值尺度一致。

## `pose_trajectory_interpolator.py` 的作用

这个文件用于对机器人末端位姿轨迹进行时间插值和 waypoint 调度。

位姿格式通常是 6 维：

```text
[x, y, z, rx, ry, rz]
```

其中：

- 前 3 维是位置
- 后 3 维是旋转向量 `rotvec`

插值方式：

- 位置用线性插值
- 旋转用 `Slerp` 球面插值

它可以：

- 查询任意时间点的 pose
- 裁剪一段轨迹
- 追加新的 waypoint
- 按最大位置速度和最大旋转速度约束运动时间

简单说，它把离散 waypoint 变成连续、可按时间查询的机器人末端轨迹。

## `model/` 文件夹的作用

`policy/DP/diffusion_policy/model` 是 DP 的模型组件库。

主要子目录：

```text
model/
  common/      通用 tensor、normalizer、rotation、scheduler 工具
  vision/      图像编码器，把 RGB 观测变成 feature
  diffusion/   diffusion policy 的去噪网络、mask、EMA 等核心模块
  bet/         Behavior Transformer / 离散动作建模相关代码
```

当前 RoboTwin DP 主要用到：

- `model/vision/multi_image_obs_encoder.py`
- `model/vision/model_getter.py`
- `model/diffusion/conditional_unet1d.py`
- `model/diffusion/mask_generator.py`
- `model/diffusion/ema_model.py`
- `model/common/normalizer.py`

## 去噪过程在哪里

在 `DiffusionUnetImagePolicy` 中，推理时真正的循环去噪过程在 `conditional_sample()` 中：

```python
for t in scheduler.timesteps:
    trajectory[condition_mask] = condition_data[condition_mask]
    model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
    trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample
```

流程是：

```text
随机噪声 action trajectory
  -> UNet 预测噪声
  -> scheduler 去掉一部分噪声
  -> 重复多步
  -> 得到最终 action trajectory
```

训练时的去噪学习过程在 `compute_loss()` 中：

1. 对真实 action trajectory 加噪。
2. 让 UNet 预测噪声。
3. 用预测噪声和真实噪声做 MSE loss。

## 从正向角度理解 Diffusion 推理

`x_t -> x_{t-1}` 不是物理时间倒流。

这里的 `t` 是 diffusion 噪声等级，不是机器人真实执行时间。

更直观的理解是：

```text
第 1 次修正随机动作草稿
第 2 次继续修正
第 3 次继续修正
...
最终得到清晰、可执行的未来动作序列
```

真实执行仍然是正向的：

```text
当前观测
  -> 生成未来动作计划 [a0, a1, a2, ...]
  -> 执行 a0
  -> 执行 a1
  -> 执行 a2
```

所以 `T -> T-1` 表示噪声程度降低，不表示现实时间倒流。

## `obs_as_global_cond` 的含义

`obs_as_global_cond` 表示观测信息是否作为 UNet 的全局条件输入。

### `obs_as_global_cond=True`

当前配置使用的是这个模式。

此时：

```text
trajectory 只包含 action
obs feature 作为 global_cond 输入 UNet
```

代码逻辑是：

```python
input_dim = action_dim
global_cond_dim = obs_feature_dim * n_obs_steps
```

因此 `LowdimMaskGenerator` 中：

```python
obs_dim = 0
```

原因是 obs 不在 trajectory 里面，不需要通过 mask 固定。

### `obs_as_global_cond=False`

此时：

```text
trajectory = [action, obs_feature]
```

obs feature 被拼进 trajectory，前 `To` 个时间步的 obs 是已知条件，所以要写进 `cond_data`：

```python
cond_data[:, :To, Da:] = nobs_features
cond_mask[:, :To, Da:] = True
```

每次去噪前都会强制写回这些已知 obs：

```python
trajectory[condition_mask] = condition_data[condition_mask]
```

这是一种 inpainting conditioning。

## `cond_data` 和 `global_cond` 的区别

`global_cond` 是模型外部的条件提示。

它不属于被 diffusion 去噪的 trajectory 本体，而是作为额外条件调制 UNet：

```text
diffusion timestep embedding + obs feature -> global feature
```

`cond_data` 是 trajectory 内部已经确定、不能被 diffusion 改掉的部分。

它必须和 `cond_mask` 配合使用：

```text
cond_mask=True 的位置：
  每次去噪前强制写回 cond_data

cond_mask=False 的位置：
  由 diffusion 生成
```

两者区别：

```text
global_cond:
  给模型看的背景信息或提示信息

cond_data:
  采样对象内部被固定住的已知值
```

当前 RoboTwin DP 配置中，obs 走 `global_cond`，action trajectory 才是 diffusion 真正生成的对象。
