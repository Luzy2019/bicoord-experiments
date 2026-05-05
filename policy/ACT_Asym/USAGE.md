# ACT_Asym 操作文档

`policy/ACT_Asym` 是从 `policy/ACT` 复制出来的独立版本，用于实现双臂非对称执行频率。原始 ACT 目录仍可按原方式使用。

## 1. 目录关系

```text
policy/ACT
  原始 ACT 行为克隆策略

policy/ACT_Asym
  ACT backbone
  + frequency projector
  + residual adapter
  + learned role/update scheduler
  + executor-level asymmetric dispatch
```

原 ACT 默认每个时间步输出完整 14 维动作：

```text
action = [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
```

ACT_Asym 仍输出统一 action chunk：

```text
A_base: [B, H, 14]
```

但会额外生成：

```text
left_update_gate:  [B, H, 1]
right_update_gate: [B, H, 1]
role_prob:         [B, 4]
```

## 2. 模型架构

整体结构：

```text
obs, qpos
  -> ACT / DETRVAE backbone
  -> A_base
  -> asymmetric adapter
  -> A_final, update gates, role prob
```

### 2.1 ACT Backbone

与原 ACT 相同：

```text
DETRVAE(image, qpos, action_chunk)
```

训练时使用 CVAE encoder 编码动作序列，推理时使用 zero latent。

输出：

```text
A_base = [a_t, a_{t+1}, ..., a_{t+H-1}]
```

### 2.2 Fixed Schedule

固定模式由人工给定 stride：

```text
left_stride=1, right_stride=5  # 左高右低
left_stride=5, right_stride=1  # 左低右高
left_stride=1, right_stride=1  # 左右同频
```

`FrequencyProjector` 会对低频臂做 hold：

```text
r0, r1, r2, r3, r4, r5
-> r0, r0, r0, r0, r0, r5
```

### 2.3 Learned Schedule

默认推荐使用 learned 模式：

```text
asym_schedule_mode=learned
asym_target_mode=original
```

它不直接学习一个固定 `stride`，而是学习每个 chunk step 是否更新：

```text
p(update_left[t])
p(update_right[t])
```

训练监督来自现有同频数据的动作变化量：

```text
动作变化大 -> update target 高
动作变化小 -> update target 低
```

这个 update 分布使用 Bernoulli KL 训练：

```text
KL(q_update_target || p_update_pred)
```

### 2.4 Role-Conditioned Scheduler

learned scheduler 同时预测粗粒度角色：

```text
0 balanced
1 left_primary
2 right_primary
3 static
```

如果推理时传入 role，则 scheduler 按 role 条件化；如果不传，则从 action dynamics 自动推断 `role_prob`。

## 3. 训练脚本

进入目录：

```bash
cd policy/ACT_Asym
```

推荐训练：

```bash
bash train_asym_bimanual.sh TASK_NAME TASK_CONFIG EXPERT_NUM SEED GPU_ID
```

例子：

```bash
bash train_asym_bimanual.sh transfer_cube default 50 0 0
```

完整参数：

```text
1  task_name
2  task_config
3  expert_data_num
4  seed
5  gpu_id
6  left_stride     可选，默认 1
7  right_stride    可选，默认 5
8  num_epochs      可选，默认 6000
9  batch_size      可选，默认 8
10 stride_choices  可选，默认 "1,1;1,5;5,1"
11 target_mode     可选，默认 original
12 schedule_mode   可选，默认 learned
```

默认脚本实际等价于：

```bash
python3 imitate_episodes.py \
  --use_asym_residual \
  --asym_schedule_mode learned \
  --asym_target_mode original \
  --asym_lambda_update_kl 1e-2 \
  --asym_lambda_update_sparse 1e-3 \
  --asym_lambda_role 1e-2
```

## 4. 训练模式选择

### 4.1 推荐：自动 learned 非对称频率

```bash
bash train_asym_bimanual.sh TASK TASK_CONFIG N SEED GPU 1 5 6000 8 "" original learned
```

适合：

```text
现有左右同频数据
希望推理时自动判断左右臂谁该高频
希望支持 role/task 条件
```

### 4.2 固定左高右低

```bash
bash train_asym_bimanual.sh TASK TASK_CONFIG N SEED GPU 1 5 6000 8 "" projected fixed
```

### 4.3 固定右高左低

```bash
bash train_asym_bimanual.sh TASK TASK_CONFIG N SEED GPU 5 1 6000 8 "" projected fixed
```

### 4.4 一个模型覆盖多种 fixed 频率

```bash
bash train_asym_bimanual.sh TASK TASK_CONFIG N SEED GPU 1 5 6000 8 "1,1;1,5;5,1" projected fixed
```

训练时每个 batch 随机采样一个 stride pair。

## 5. 测试脚本

进入目录：

```bash
cd policy/ACT_Asym
```

运行：

```bash
bash eval.sh TASK_NAME TASK_CONFIG CKPT_SETTING EXPERT_NUM SEED GPU_ID
```

例子：

```bash
bash eval.sh transfer_cube default default 50 0 0
```

`eval.sh` 会调用：

```bash
python script/eval_policy.py \
  --config policy/ACT_Asym/deploy_policy.yml \
  --ckpt_dir policy/ACT_Asym/act_ckpt/asym-act-${task_name}/${ckpt_setting}-${expert_data_num}-L1-R5
```

如果你的 checkpoint 路径不同，需要修改 `eval.sh` 里的 `--ckpt_dir`。

## 6. 推理使用方式

普通调用仍兼容原 ACT：

```python
action = model.get_action(obs)
```

如果需要左右臂分发信息：

```python
packet = model.get_action(obs, return_packet=True)
```

返回：

```python
{
    "action": full_14d_action,
    "left_action": left_7d_action,
    "right_action": right_7d_action,
    "left_update": bool,
    "right_update": bool,
    "role_prob": optional_role_distribution,
}
```

## 7. 传入任务角色

可以在 obs 中传：

```python
obs["asym_role"] = "left_primary"
```

或：

```python
obs["asym_role_id"] = 1
```

支持的 role：

```text
balanced / equal
left_primary / left_high / left_operate
right_primary / right_high / right_operate
static / hold
```

如果环境提供：

```python
TASK_ENV.get_asym_role()
```

`deploy_policy.py` 会自动读取。

## 8. 左右臂分开发送

`deploy_policy.dispatch_asymmetric_action()` 会按优先级选择执行接口：

```text
1. TASK_ENV.take_action_asym(left_action, right_action, left_update, right_update)
2. TASK_ENV.take_left_action(...) + TASK_ENV.take_right_action(...)
3. TASK_ENV.take_action(full_action)
```

只有前两种接口才是真正的通信/控制频率非对称。第三种 fallback 只是动作值 hold，仍每步发送完整 14 维动作。

## 9. 关键配置

`deploy_policy.yml` 中：

```yaml
use_asym_residual: false
asym_schedule_mode: learned
asym_target_mode: original
asym_hard_inference: true
asym_update_threshold: 0.5
asym_num_roles: 4
asym_use_role_condition: true
```

部署 ACT_Asym checkpoint 时，需要确保覆盖：

```yaml
use_asym_residual: true
```

否则会退化为普通 ACT。

## 10. 损失函数

总损失：

```text
L = L1(A_final, A_target)
  + kl_weight * L_vae_kl
  + lambda_res * ||delta||^2
  + lambda_gate * ||gate||_1
  + lambda_smooth * smooth(A_final)
  + lambda_freq * low_freq_consistency
  + lambda_update_kl * KL(update_target || update_pred)
  + lambda_update_sparse * sparsity(update_pred)
  + lambda_role * CE(role_pred, pseudo_role)
```

learned 模式中：

```text
A_target = 原始同频动作
```

fixed + projected 模式中：

```text
A_target = hold-projected 动作
```

## 11. 重要注意

1. 现有同频数据可以直接训练 learned 模式。
2. learned 模式学的是 update gate，不是离散固定 stride。
3. 如果任务明确知道左/右主操作臂，推理时最好传入 role。
4. 如果不传 role，模型会从 action chunk 自动估计 role。
5. 真正的执行频率非对称需要环境支持左右臂分开发送 API。
