# DP_AT: Action-Time Diffusion Policy

这个分支从 `policy/DP` 轻量复制而来，实现的是最终讨论的 **不使用 alpha、不使用显式门控** 的方案：

```text
Y = (A_L, A_R, t_L, t_R)
```

模型直接对动作-时间联合 trace 做 diffusion generation。动作通道照常送给环境，时间通道表示左右臂 action chunk 的执行时间结构，可用于 compact scheduling、WM/energy rerank 和后续蒸馏。

## 核心文件

- `diffusion_policy/policy/action_time_diffusion_unet_image_policy.py`
- `diffusion_policy/config/robot_dp_at_16.yaml`
- `train_at.sh`

## 训练目标

```text
L = L_DDPM(Y)
  + lambda_energy * L_energy(Y)
  + lambda_world * L_world(z_t, Y_t -> z_{t+1})
  + lambda_mono * L_monotonic(t_L, t_R)
```

其中：

- `L_DDPM(Y)` 学习 demo 或 pseudo-label 的动作-时间联合分布。
- `L_energy` 学习 demo-like trace feasibility，推理时和 time cost 一起 rerank。
- `L_world` 是轻量 latent world model，用于可行性 verifier。
- `L_monotonic` 约束时间通道随 chunk 单调。

## 没有 compact label 时怎么办

如果 zarr 没有 `time_trace`，dataset 会自动使用线性 demo 时间：

```text
t_L = t_R = linspace(0, 1, H)
```

这只是 base model。真正的 compact 学习路线是：

```text
1. 用 demo 训练 DP_AT base policy
2. 在 sim/WM 中生成 action-time candidates
3. 用 success / collision / uncertainty / makespan 打分
4. 把高分 compact trace 写回 zarr 的 data/time_trace
5. 再训练 DP_AT，蒸馏 compact execution trace
```

也就是说，紧凑性不是从原 demo 标签里凭空学出来，而是由 **成功约束下最小化 makespan** 的搜索目标产生。

## 训练

```bash
cd policy/DP_AT
bash train_at.sh stack_bowls demo_clean 50 42 0
```

增加推理候选 rerank：

```bash
bash train_at.sh stack_bowls demo_clean 50 42 0 4
```

## 输出

`predict_action` 返回：

- `action`: 现有环境可执行动作。
- `action_pred`: 完整动作 chunk。
- `time_trace`: 生成的 `(t_L, t_R)`。
- `candidate_scores`: 多候选 energy + compactness 分数。

=====

我认可你的质疑，而且我会更直接地说：**`alpha * velocity` 和显式门控更像一个可跑的工程 baseline，不像最优雅、最“生成模型原生”的方案。**

它们的问题是：

1. **`alpha * v` 有尺度不可辨识**
   
   ```text
   alpha * v = (alpha / c) * (c v)
   ```
   
   所以模型到底是在学“方向”，还是在乱调速度尺度，不清楚。这个确实有点硬。

2. **门控 `g` 有人为结构假设**
   
   ```text
   v = v_local + g * residual
   ```
   
   这很可解释，但也会被质疑：为什么一定是这个分解？为什么不是模型自己学出耦合结构？

3. **它不像纯 EBM / Flow 那么自然**
   
   EBM 本来就可以直接定义：
   
   ```text
   E(A_L, A_R, schedule | O)
   ```
   
   Flow Matching 也可以直接生成：
   
   ```text
   (A_L, A_R, timing_L, timing_R)
   ```
   
   不一定需要手动乘一个 `alpha` 或写一个门控。

我现在更倾向于把 `alpha/gate` 定位成：**第一版诊断工具，而不是最终方法。**

更优雅的设计应该是：**直接把“紧凑执行”作为生成变量的一部分。**

也就是不要写：

```text
action = original_action warped by alpha
```

而是直接生成：

```text
Y = (A_L, A_R, τ_L, τ_R)
```

其中：

- `A_L, A_R` 是左右臂动作轨迹；
- `τ_L, τ_R` 是左右臂动作的执行时间戳 / duration / event schedule；
- 紧凑性由 `makespan = max(τ_L, τ_R)` 衡量。

**EBM 版本最自然：**

```text
E_total(Y | O)
= E_demo(A_L, A_R | O)
+ E_WM_failure(Y, O)
+ E_collision(Y)
+ E_precedence(Y)
+ E_dyn(Y)
+ λ_time * makespan(Y)
```

然后直接优化或采样：

```text
Y* = argmin E_total(Y | O)
```

这里没有 `alpha`，也没有显式门控。耦合关系由能量项自己表达：该同步就同步，该并行就并行。

**Flow Matching 版本也可以更干净：**

训练一个 flow：

```text
noise -> (A_L, A_R, τ_L, τ_R)
```

如果没有紧凑轨迹数据，就先用 base policy / demo 产生候选，再用 WM + EBM + time cost 评分，选出高分 compact candidates，最后蒸馏成新的 flow：

```text
p_compact(Y | O) ∝ p_demo(Y | O) exp(-E_WM(Y,O) - λ_time makespan)
```

然后训练：

```text
v_compact(Y_t, t, O)
```

这比 `alpha * velocity` 更像真正的生成模型：**新的向量场直接生成紧凑轨迹，而不是后处理原轨迹。**

所以我现在的判断是：

- `alpha`：适合作为早期实验、可解释 speed baseline。
- `gate`：适合作为分析耦合强度的可视化工具。
- 最终论文方法：更推荐 **augmented action-time generation**。
- 最强形式：**Flow generates candidates，EBM/WM scores compactness and feasibility，再蒸馏回 Flow。**

一句话：你这个直觉是对的。`alpha/gate` 有点“工程化插头”的味道；如果想做得更漂亮，应该让 Flow/EBM **直接在动作-时间联合空间里生成紧凑双臂轨迹**。

是的，可以不用 `alpha` 和显式门控，而且我觉得更优雅的方向是：**不要让模型“加速一条已有轨迹”，而是让模型直接生成“动作 + 时间结构”**。

也就是把问题从：

```text
原轨迹 A 乘一个速度系数 alpha
```

改成：

```text
直接生成 execution trace:
Y = (A_L, A_R, t_L, t_R)
```

其中 `t_L, t_R` 是左右臂每个动作/事件的执行时间戳、duration 或 start-end interval。这样“加速”不是一个外接模块，而是生成结果本身。

我觉得有几种优雅方案。

**1. Action-Time Joint Flow**

让 Flow Matching 直接生成：

```text
noise -> (left_action_chunk, right_action_chunk, left_time, right_time)
```

训练时 demo 只有普通时间结构，确实没有 compact 轨迹。所以不能只靠 supervised FM。要加一个 WM/energy 目标，把目标分布改成：

```text
p_compact(Y | O) ∝ p_demo(Y | O) * exp(
    reward_success(Y)
    - λ_time * makespan(Y)
    - λ_risk * risk(Y)
)
```

然后用 WM 评估候选 `Y` 是否成功、安全、可行。高分 compact candidates 再蒸馏回 flow。

这比 `alpha * v` 干净，因为 flow 学到的是新的目标分布，不是手动改速度。

**2. EBM Over Execution Trace**

直接定义能量：

```text
E(Y | O)
= E_demo_path(A_L, A_R | O)
+ E_WM_failure(Y, O)
+ E_collision(Y)
+ E_precedence(Y)
+ E_dyn(Y)
+ λ_time * makespan(Y)
```

推理时：

```text
Y* = argmin E(Y | O)
```

这里 `makespan` 就是总执行时长。模型会自然倾向于压缩时间，但如果压缩导致失败、碰撞、顺序错乱，能量会升高。

这个方案最像“真正的能量模型”：低能量区域就是理想 compact 轨迹区域。

**3. Latent Dependency Graph**

这是我觉得最有研究味的方案。

把双臂任务拆成事件节点：

```text
L_pick, L_place, R_pick, R_place, shared_contact, release...
```

模型学习一个 latent dependency graph：

```text
edge i -> j 表示 j 必须等 i 完成
无边表示可以并行
```

然后 compact execution 不是模型硬学出来的，而是由调度器自然得到：

```text
在满足 dependency graph 的前提下最小化总时间
```

这很漂亮，因为“耦合不紧凑时可以并行”被表达成：

```text
左右臂事件之间没有强依赖边
```

而不是 `gate` 或 `alpha`。

**4. Flow Proposal + WM Search + Distillation**

这是最实用的版本：

```text
1. Base flow 生成普通动作候选
2. 构造多个 action-time candidates
3. WM 评估 success / collision / uncertainty / goal distance
4. 加 time cost 选 compact candidate
5. 把选出来的 compact candidate 蒸馏回 flow
```

训练目标变成：

```text
L = L_FM(Y_compact)
```

其中 `Y_compact` 不是人工标注，而是 WM/search 生成的 pseudo-optimal trace。

这就是让模型“学”出来加速：不是靠标签，而是靠 **search -> verify -> distill**。

我的推荐路线是：

```text
第一版：EBM/WM 评分 action-time candidates
第二版：把高分 compact candidates 蒸馏进 Flow
第三版：学习 latent dependency graph，提升可解释性
```

一句话总结：  
**不要学 alpha，不要学 gate；学一个动作-时间联合分布，或者学一个任务依赖图。紧凑性来自 “成功约束下最小化执行时间” 这个目标，而不是来自 demo 标签。**

我更倾向于第二种：**先学一个稳定成功的 base model，再像 model-based RL / policy improvement 一样，在仿真或世界模型里不断搜索、验证、蒸馏紧凑轨迹。**

原因很简单：你的 demo 里没有“紧凑轨迹”，所以如果一开始就在监督训练里学加速，模型其实没有可靠 target。它只能靠一些 proxy，比如时间惩罚、速度惩罚、风险项，容易学成很硬的启发式。

更稳的路线是：

```text
1. Demo -> 训练成功率高的 base Flow/DP
2. base policy 作为行为先验 / 安全下限
3. 在 sim 或 WM 中生成 action-time candidates
4. 用 reward/energy 评估：成功、安全、碰撞、时间、动力学、OOD
5. 选出 compact successful traces
6. 蒸馏回新的 Flow / EBM policy
```

也就是：

```text
p_compact(Y|O) ∝ p_base(Y|O) * exp(Q_WM(Y,O) - λ_time makespan(Y))
```

其中 `Y` 是动作加时间结构：

```text
Y = (A_L, A_R, t_L, t_R)
```

这样“加速”不是训练时凭空想象出来的，而是通过反事实交互得到的。

我不建议从零端到端学加速，主要有三个问题：

1. **没有监督信号**：demo 只证明慢轨迹成功，不证明快轨迹成功。
2. **容易破坏成功率**：时间越压缩，越容易碰撞、错过接触、打乱顺序。
3. **目标冲突**：BC 想贴近 demo，compactness 想偏离 demo 的时间结构。

所以最好的说法不是“训练时学加速”或“RL 后处理”二选一，而是：

**先 imitation 保成功，再 model-based improvement 找紧凑，最后 distill 成一个直接生成紧凑轨迹的模型。**

最小可行实验我会这样做：

```text
Base Flow/DP
-> 采样 K 个动作候选
-> 对每个候选生成多种左右臂时间排布
-> sim/WM rollout 打分
-> reward = success - λ_time * makespan - penalties
-> 保存高分 compact trace
-> 训练 Action-Time Flow
```

如果仿真环境可靠，优先用仿真交互；如果仿真慢，再训练 WM 做快速筛选，但最终最好还是用真实 sim 验证 compact candidate。

一句话：**紧凑轨迹不是直接从 demo 学出来的，而是从“成功模型 + 时间优化 + 环境/世界模型验证”里进化出来的。**

已按最后讨论的方案完成两套独立实现：

- DP 版：`policy/DP_AT`
- FM 版：`policy/FM_AT`

这两个分支都不是 `alpha`/显式 gate 方案，而是 **action-time joint generation**：

```text
Y = (A_L, A_R, t_L, t_R)
```

核心文件：

- [DP action-time policy](/home/lzy/code/BiCoord-Bench/policy/DP_AT/diffusion_policy/policy/action_time_diffusion_unet_image_policy.py:116)
- [FM action-time policy](/home/lzy/code/BiCoord-Bench/policy/FM_AT/flow_matching_policy/policy/action_time_flow_unet_image_policy.py:109)
- [DP config](/home/lzy/code/BiCoord-Bench/policy/DP_AT/diffusion_policy/config/robot_dp_at_16.yaml:1)
- [FM config](/home/lzy/code/BiCoord-Bench/policy/FM_AT/flow_matching_policy/config/robot_fm_at_16.yaml:1)
- [DP README](/home/lzy/code/BiCoord-Bench/policy/DP_AT/README_AT.md:1)
- [FM README](/home/lzy/code/BiCoord-Bench/policy/FM_AT/README_AT.md:1)

训练入口：

```bash
cd policy/DP_AT
bash train_at.sh stack_bowls demo_clean 50 42 0
```

```bash
cd policy/FM_AT
bash train_at.sh stack_bowls demo_clean 50 100 0
```

还加了 `time_trace` 蒸馏接口：如果 zarr 里有 `data/time_trace`，dataset 会自动读取；没有的话默认使用线性 demo 时间。这样可以支持后续“base policy -> sim/WM 搜索 compact trace -> 写回 time_trace -> 蒸馏”的闭环。

验证：两个分支全目录 `compileall` 通过，`train_at.sh/process_data.sh/eval.sh` shell 语法检查通过。当前系统 Python 没有 `torch`，所以没法在这里跑真实 forward/backward。目标已完成，用时约 10 分半。