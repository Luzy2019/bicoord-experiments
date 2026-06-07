# 当前会话原始对话记录

来源：当前 DP 代码讨论会话 transcript。

## 1. 用户

<attached_files>

<code_selection path="/home/lzy/code/RoboTwin/policy/DP/dp_model.py" lines="33-52">
L33:    def get_policy(self, checkpoint, output_dir, device):
L34:        # load checkpoint
L35:        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
L36:        cfg = payload["cfg"]
L37:        cls = hydra.utils.get_class(cfg._target_)
L38:        workspace = cls(cfg, output_dir=output_dir)
L39:        workspace: RobotWorkspace
L40:        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
L41:
L42:        # get policy from workspace
L43:        policy = workspace.model
L44:        if cfg.training.use_ema:
L45:            policy = workspace.ema_model
L46:
L47:        device = torch.device(device)
L48:        policy.to(device)
L49:        policy.eval()
L50:
L51:        return policy
L52:
</code_selection>

</attached_files>
<timestamp>Tuesday, Apr 28, 2026, 11:20 AM (UTC+8)</timestamp>
<user_query>
@policy/DP/dp_model.py:33-52 解释一下这段代码
</user_query>

---

## 2. 助手

我先看一下这个文件周边定义，确认 `get_policy` 所在类和调用语境，再解释这段加载逻辑。

**Explaining code in Chinese**

I need to provide an answer in Chinese, which is important for clarity. I should explain the code effectively. It might help to read the full file to get comprehensive context, but perhaps a selection of key points will be sufficient. I'll focus on making sure the explanation is clear and understandable in Chinese, ensuring I convey the necessary details without overwhelming the user.

---

## 3. 助手

这段 `get_policy` 的作用是：**从训练好的 checkpoint 中恢复 Diffusion Policy 模型，并返回一个可用于推理的 policy 网络**。

```33:51:policy/DP/dp_model.py
    def get_policy(self, checkpoint, output_dir, device):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy
```

逐步解释：

1. `torch.load(...)`  
   从 `checkpoint` 文件里加载训练保存的内容。这里用 `dill` 作为反序列化模块，说明 checkpoint 里可能包含一些普通 `pickle` 不好处理的 Python 对象。

2. `cfg = payload["cfg"]`  
   取出训练时保存的 Hydra 配置。这个配置里记录了模型、workspace、训练参数等信息。

3. `cls = hydra.utils.get_class(cfg._target_)`  
   根据配置里的 `_target_` 字符串动态找到对应的 workspace 类。也就是说，代码不直接写死某个类，而是由 checkpoint 中的配置决定要实例化哪个 workspace。

4. `workspace = cls(cfg, output_dir=output_dir)`  
   创建 workspace 实例。workspace 通常负责管理模型、优化器、EMA 模型、训练状态等。

5. `workspace.load_payload(payload, ...)`  
   把 checkpoint 中保存的权重和状态加载回 workspace。

6. `policy = workspace.model`  
   默认使用 workspace 里的主模型作为推理策略。

7. `if cfg.training.use_ema:`  
   如果训练时启用了 EMA，则使用 `workspace.ema_model`。EMA 是训练过程中模型参数的滑动平均版本，通常推理效果更稳定。

8. `policy.to(device)`  
   把模型移动到指定设备，比如这里调用处传的是 `"cuda:0"`。

9. `policy.eval()`  
   切换到推理模式，关闭 dropout、固定 batchnorm 等训练行为。

10. `return policy`  
   返回已经加载好权重、移动到设备、设置为 eval 模式的策略模型。

简单说，这个函数就是：**读取 checkpoint → 根据配置重建 workspace → 恢复模型权重 → 选择普通模型或 EMA 模型 → 放到 GPU → 设置推理模式 → 返回 policy**。

---

## 4. 用户

<timestamp>Tuesday, Apr 28, 2026, 11:24 AM (UTC+8)</timestamp>
<user_query>
workspace是什么？是trainner吗
</user_query>

---

## 5. 助手

我看一下 `RobotWorkspace` 的定义，这样可以准确说明它和 trainer 的关系。

---

## 6. 助手

`workspace` 不是严格意义上的 `trainer`，但它**承担了训练管理器/实验容器**的角色。

在这个项目里，`RobotWorkspace` 里会保存和管理这些东西：

```32:58:policy/DP/diffusion_policy/workspace/robotworkspace.py
class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
```

可以把它理解成一个“训练工作区”：

- `self.model`：真正的策略模型，也就是 Diffusion Policy 网络
- `self.ema_model`：EMA 版本模型，用于更稳定的推理
- `self.optimizer`：训练用优化器
- `self.global_step / self.epoch`：训练进度
- `run()`：真正执行训练流程的方法
- `load_payload()`：从 checkpoint 恢复模型和训练状态

所以它和 trainer 的关系是：

**如果你说的 trainer 是“负责训练循环的对象”，那 `RobotWorkspace` 就很像 trainer。**  
但更准确地说，它比 trainer 更宽一点，是一个 workspace：不仅包含训练循环，还包含模型创建、优化器、checkpoint 加载保存、EMA、日志、数据集配置等实验状态。

在 `dp_model.py` 这里用它时，并不是为了训练，而是为了：

```python
workspace = cls(cfg, output_dir=output_dir)
workspace.load_payload(payload, ...)
policy = workspace.model
```

也就是：**借助 workspace 按训练时的配置重建模型结构，然后加载 checkpoint 权重，最后取出 policy 用于部署推理。**

---

## 7. 用户

<timestamp>Tuesday, Apr 28, 2026, 11:33 AM (UTC+8)</timestamp>
<user_query>
@policy/DP 在这个DP实现中，使用的仍然是unet对把，没有用transformer实现diffusion
</user_query>

---

## 8. 助手

我会快速核对 `policy/DP` 的配置和 policy 类，确认实际实例化的模型骨干是不是 UNet，以及有没有 Transformer 相关实现被接入。

---

## 9. 助手

已经看到配置里 `cfg.policy` 指向 `DiffusionUnetImagePolicy`，这个类内部实例化的是 `ConditionalUnet1D`；仓库里虽有 Transformer 模块，但当前 DP 配置没有把它作为 diffusion policy 使用。

---

## 10. 助手

对，你的判断是对的：`policy/DP` 这个实现里，当前使用的是 **UNet 版 diffusion policy**，不是 Transformer 版 diffusion。

关键证据有两个。

配置里 `policy._target_` 指向的是 `DiffusionUnetImagePolicy`：

```21:23:policy/DP/diffusion_policy/config/robot_dp_14.yaml
policy:
  _target_: diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy
```

而 `DiffusionUnetImagePolicy` 内部创建的 denoising model 是 `ConditionalUnet1D`：

```45:61:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
```

所以这里的结构大致是：

**图像观测 → ResNet encoder 提特征 → 条件输入给 `ConditionalUnet1D` → DDPM scheduler 做 diffusion 去噪 → 输出 action 序列。**

仓库里确实有 Transformer 相关代码，比如 `diffusion_policy/model/diffusion/transformer_for_diffusion.py`，但从当前 `robot_dp_14.yaml` / `robot_dp_16.yaml` 配置看，它没有被用作当前 DP policy 的主干。这里 diffusion 的噪声预测网络是 **1D Conditional UNet**。

---

## 11. 用户

<timestamp>Tuesday, Apr 28, 2026, 11:36 AM (UTC+8)</timestamp>
<user_query>
@policy/DP/diffusion_policy/common/checkpoint_util.py 这个的作用是什么
</user_query>

---

## 12. 助手

我先读一下 `checkpoint_util.py` 的完整内容，再按函数说明它在 checkpoint 保存/筛选里的作用。

---

## 13. 助手

这个文件的作用是：**管理只保留 Top-K 个最好的 checkpoint**，避免训练过程中保存太多模型文件。

核心类是 `TopKCheckpointManager`。

它的逻辑是：每次训练到某个 epoch，传入当前指标，比如 `train_loss`、`val_loss`、`success_rate`，它判断这个 checkpoint 是否值得保存。如果值得保存，就返回保存路径；如果不值得保存，就返回 `None`。

主要参数：

- `save_dir`：checkpoint 保存目录
- `monitor_key`：用哪个指标判断好坏，比如 `"val_loss"`
- `mode`：指标越大越好还是越小越好，`"max"` 或 `"min"`
- `k`：最多保留几个 checkpoint
- `format_str`：checkpoint 文件名格式

关键函数是：

```25:61:policy/DP/diffusion_policy/common/checkpoint_util.py
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]
```

举个例子，如果配置是：

```python
TopKCheckpointManager(
    save_dir="checkpoints",
    monitor_key="val_loss",
    mode="min",
    k=3,
)
```

那么它会最多保留 **val_loss 最小的 3 个 checkpoint**。

如果来了一个新的 checkpoint：

- 当前不足 3 个：直接允许保存
- 已经有 3 个：比较新指标
- 如果新 `val_loss` 比当前最差的还好：删除最差的，保存新的
- 如果不够好：返回 `None`，不保存

所以它不是负责“真正保存 checkpoint”的类，而是负责判断：

**这个 checkpoint 要不要保存？如果要保存，应该保存到哪个路径？如果 Top-K 满了，要删掉哪个旧 checkpoint？**

---

## 14. 用户

<timestamp>Tuesday, Apr 28, 2026, 11:37 AM (UTC+8)</timestamp>
<user_query>
@policy/DP/diffusion_policy/common/cv2_util.py 这个文件的作用是什么，你帮我在@policy/DP/diffusion_policy/common/__test__/cv2_util_test.py 文件中，写一个测试吧，类似于单元测试，我想先看看效果
</user_query>

---

## 15. 助手

我先看一下 `cv2_util.py` 的函数内容和现有测试文件状态，然后给你补一个小型可运行的测试用例，方便直接观察输出效果。

**Preparing response in Chinese**

I need to respond in Chinese, so I'll focus on that task. I should edit the test and inspect the files as necessary. It seems like there might not be a need for a plan mode right now, which could simplify things. I wonder if the user has already given an update before I start the work—it's important to have that context! I'll definitely use the ReadFile function for this.

---

## 16. 助手

这个工具文件主要做图像绘制、缩放裁剪和多相机拼图布局计算；我准备写一个 `unittest` 风格测试，覆盖这些函数，并在临时目录里生成一张 demo 图验证绘制效果。

**Planning test setup**

I need to write a test, and I’m considering whether to use pytest or a simple script. The user wants something similar to unit tests, possibly including unittest with image-writing features. I should remember to import the necessary paths correctly, especially depending on where the test is executed. It might be better to adjust the sys.path to ensure everything works smoothly. 

I’ll cover modifying images in my tests but avoid saving generated images directly in the repo.

---

## 17. 助手

测试文件已经补上了。接下来我跑一下这个单测，确认导入路径和 OpenCV 行为都正常。

**Evaluating patch procedures**

I’m considering if I should apply a patch to add a file in an existing empty workspace. It indicates that it has been updated. I might need to check the necessary imports, like cv2, and calculate the system path correctly. The structure seems to lead to the DP root, which is good, but the test file is currently empty. I should run the test and review shell instructions for any relevant terminal commands needed before proceeding.

---

## 18. 助手

当前环境里没有 `python` 命令，只有 `python3`；我用 `python3` 再跑一次。

---

## 19. 用户

<attached_files>

<terminal_selection title="Terminal" path="/home/lzy/.cursor/projects/home-lzy-code-RoboTwin/terminals/1.txt" lines="32-38">
(RoboTwin) lzy@lzy-MS-7E06:~/code/RoboTwin$ python '/home/lzy/code/RoboTwin/policy/DP/diffusion_policy/common/__test__/cv2_util_test.py'
...
----------------------------------------------------------------------
Ran 3 tests in 0.115s

OK
</terminal_selection>

</attached_files>
<timestamp>Tuesday, Apr 28, 2026, 11:39 AM (UTC+8)</timestamp>
<user_query>
@/home/lzy/.cursor/projects/home-lzy-code-RoboTwin/terminals/1.txt:32-38 效果是什么？怎么就只有控制台的输出呢
</user_query>

---

## 20. 助手

现在这个测试的“效果”主要是验证函数有没有正常工作，所以默认只会看到控制台输出：

```text
Ran 3 tests in 0.115s
OK
```

意思是 3 个测试都通过了：

- `draw_reticle` 和 `draw_text` 确实修改了图片像素
- `get_image_transform` 能把图片 resize/crop 到目标尺寸，并做 BGR→RGB 通道翻转
- `optimal_row_cols` 能算出多相机画面布局

你之所以没看到图片，是因为我测试里用的是：

```python
with tempfile.TemporaryDirectory() as tmp_dir:
```

这个目录是临时目录，测试结束后会自动删除。所以它只是验证“图片能写出来”，不会留下可查看的图片文件。

如果你想**实际看到画出来的效果图**，可以把保存路径改成固定路径，例如：

```python
output_path = "/home/lzy/code/RoboTwin/policy/DP/diffusion_policy/common/__test__/cv2_util_demo.png"
cv2.imwrite(output_path, img)
print(f"saved demo image to: {output_path}")
```

这样运行后就会在 `__test__` 目录下生成 `cv2_util_demo.png`，你可以直接打开看十字准星和文字效果。

---

## 21. 用户

<timestamp>Tuesday, Apr 28, 2026, 11:40 AM (UTC+8)</timestamp>
<user_query>
好的，你帮我添加可视化效果吧
</user_query>

---

## 22. 助手

我来把测试里的临时图片改成固定输出文件，并打印保存路径，这样运行后你可以直接打开生成的 `png` 看效果。

**Editing file for response**

I see that I need to edit the file and respond in Chinese. First, I should probably read the current file to understand its contents. Then, I can make the necessary edits or patches. It’s also a good idea to run a test afterward to ensure everything functions correctly. I definitely should wait for the user to provide updates before I proceed with any edits. Let's work through this step-by-step!

---

## 23. 助手

我会去掉 `tempfile`，改为写到当前测试目录下的 `cv2_util_demo.png`，并在测试输出里打印文件路径。

---

## 24. 助手

改好了。现在我运行一次测试，顺便生成这张可视化 demo 图片。

---

## 25. 用户

<timestamp>Tuesday, Apr 28, 2026, 11:41 AM (UTC+8)</timestamp>
<user_query>
这个看不出这个模块的作用呢，你帮我找一个有意义的图，可以从网上找并下载下来后，处理 该图片
</user_query>

---

## 26. 助手

我会把 demo 换成真实图片流程：从 OpenCV 官方示例图下载一张图，保存原图，再用 `cv2_util.py` 做 resize/crop、画准星、加文字，输出处理后的图片，这样能直观看到这个模块的用途。

**Finding and processing images**

I need to help the user find a meaningful image online to download and process. I'll consider using web search or a shell command like curl, but I need to check if downloading from URLs is allowed. Using a stable URL, like the OpenCV logo or a classic image like "lena.jpg," is a good idea for testing image processing. I should ensure that if the network's unavailable, we can skip the test. Adjusting my approach, I'll download this sample image and apply transformations.
**Refining image download process**

I think I should use `urllib.request` for downloads, but I’m aware that tests relying on network availability can be fragile. I could write a script to include a download-if-needed function and skip visual tests if the network is down. The user wants actual effects, and my previous Python shell run might have been interrupted due to environmental issues. The terminal command worked using a conda environment, but the shell Python may hang or start incorrectly. I need to patch the current file, potentially using fixed URLs and comments while ensuring that any download failures are handled properly.
**Creating download and process helpers**

I think I should write a helper function called `_download_demo_image`. I’ll use URL constants to simplify the process, saving images as `cv2_util_source.jpg` and `cv2_util_processed.png`. In my tests, if the source image isn’t available, I can implement a download step. The goal here is to ensure that I can process a natural image effectively. This should help streamline the workflow and make sure everything runs smoothly for the user!

---

## 27. 助手

我现在修改测试文件，把原来的黑底测试图保留为基础单测，再新增一个“真实图片可视化”测试，生成原图和处理图两个文件。

---

## 28. 助手

文件已经改好。接下来我试运行一次；如果网络能访问 GitHub，会生成 `cv2_util_source.jpg` 和 `cv2_util_processed.png`。

---

## 29. 用户

<timestamp>Tuesday, Apr 28, 2026, 11:43 AM (UTC+8)</timestamp>
<user_query>
@policy/DP/diffusion_policy/common/normalize_util.py 这个文件中的代码的作用是什么
</user_query>

---

## 30. 助手

我先读一下 `normalize_util.py` 的完整内容，再按函数分组解释它在 DP 数据归一化里的作用。

---

## 31. 助手

`normalize_util.py` 的作用是：**根据数据统计量构造归一化器，把 action、图像或其他连续数据变换到模型更容易学习的范围里**。在 Diffusion Policy 里，这很重要，因为模型通常希望输入/输出都在稳定范围，比如 `[-1, 1]`。

最基础的是这个函数：

```10:21:policy/DP/diffusion_policy/common/normalize_util.py
def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stat)
```

它做的是典型的 min-max normalization：

```text
normalized = raw * scale + offset
```

默认会把原始数据按每个维度的 `min/max` 映射到 `[-1, 1]`。

这个文件里主要有几类函数：

- `get_range_normalizer_from_stat`：把普通连续数据按统计范围归一化到 `[-1, 1]`
- `get_image_range_normalizer`：专门给图像用，把 `[0, 1]` 的像素映射到 `[-1, 1]`
- `get_identity_normalizer_from_stat`：不做归一化，`scale=1, offset=0`
- `array_to_stats`：从一批 numpy 数据里计算 `min/max/mean/std`
- `robomimic_abs_action_*_normalizer_from_stat`：针对 robomimic 风格的机器人动作做特殊归一化

机器人动作这里比较特殊。比如单臂 action 可能是：

```text
[x, y, z, rotation..., gripper...]
```

代码会把 action 拆成不同部分：

```42:44:policy/DP/diffusion_policy/common/normalize_util.py
def robomimic_abs_action_normalizer_from_stat(stat, rotation_transformer):
    result = dict_apply_split(stat, lambda x: {"pos": x[..., :3], "rot": x[..., 3:6], "gripper": x[..., 6:]})
```

其中：

- `pos`：位置，按真实数据的 `min/max` 归一化到 `[-1, 1]`
- `rot`：旋转，不直接用原始欧拉角范围，而是先通过 `rotation_transformer` 转成另一种旋转表示
- `gripper`：夹爪，基本保持 identity，因为它可能本来就是离散或固定范围值

双臂版本 `robomimic_abs_action_only_dual_arm_normalizer_from_stat` 也是类似逻辑，只是把 action 切成：

```text
左臂 pos + 左臂 other + 右臂 pos + 右臂 other
```

简单说，这个文件就是 DP 训练前的数据预处理工具：**根据数据集统计量创建 normalizer，保证模型训练和推理时 action/image 的数值尺度一致。**

---

## 32. 用户

<timestamp>Tuesday, Apr 28, 2026, 11:44 AM (UTC+8)</timestamp>
<user_query>
@policy/DP/diffusion_policy/common/pose_trajectory_interpolator.py 这个文件的作用是什么
</user_query>

---

## 33. 助手

我先读一下 `pose_trajectory_interpolator.py`，再结合函数结构解释它如何对位姿轨迹按时间插值、裁剪和追加 waypoint。

---

## 34. 助手

这个文件的作用是：**对机器人末端位姿轨迹做时间插值和 waypoint 调度**。

这里的 `pose` 是 6 维：

```text
[x, y, z, rx, ry, rz]
```

前 3 维是位置，后 3 维是旋转向量 `rotvec`。

核心类是 `PoseTrajectoryInterpolator`。它接收一组时间点和对应位姿：

```python
times = [0.0, 1.0, 2.0]
poses = [
    [x0, y0, z0, rx0, ry0, rz0],
    [x1, y1, z1, rx1, ry1, rz1],
    [x2, y2, z2, rx2, ry2, rz2],
]
```

然后你可以在任意时间查询中间位姿：

```python
pose_at_0_5 = interpolator(0.5)
```

它内部做了两种插值：

```43:47:policy/DP/diffusion_policy/common/pose_trajectory_interpolator.py
            pos = poses[:, :3]
            rot = st.Rotation.from_rotvec(poses[:, 3:])

            self.pos_interp = si.interp1d(times, pos, axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)
```

也就是：

- 位置 `x/y/z`：用线性插值
- 旋转 `rx/ry/rz`：用 `Slerp` 球面插值，避免旋转插值不自然

几个重要函数：

- `rotation_distance(a, b)`：计算两个旋转之间的角度距离
- `pose_distance(start_pose, end_pose)`：计算两个 pose 的位置距离和旋转距离
- `trim(start_t, end_t)`：截取一段轨迹，并自动插值出边界点
- `drive_to_waypoint(...)`：从当前时间出发，开到一个目标 pose
- `schedule_waypoint(...)`：往已有轨迹里安排一个新的 waypoint，同时考虑当前时间、上一个 waypoint 和速度限制
- `__call__(t)`：查询某个时间点对应的 pose

它比较关键的用途是控制机器人运动时，让离散的目标点变成连续、平滑、按时间可查询的轨迹。例如 policy 输出未来几个 action waypoint 后，底层控制器可以用这个类在每个控制周期查询当前应该到达的末端位姿。

简单说，这个文件就是：

**把一串离散的机器人末端位姿点，变成可按时间查询的连续轨迹，并且支持裁剪、追加目标点和速度约束。**

---

## 35. 用户

<timestamp>Tuesday, Apr 28, 2026, 11:54 AM (UTC+8)</timestamp>
<user_query>
@policy/DP/diffusion_policy/model/diffusion/transformer_for_diffusion.py 但是我又看到了关于transformer的diffusion，这个和unet diffusion的区别是什么
</user_query>

---

## 36. 助手

我会对照 `transformer_for_diffusion.py` 和当前实际使用的 `diffusion_unet_image_policy.py`，解释两者都是 diffusion 的噪声预测器，但网络骨干、条件方式和适用场景不同。

---

## 37. 助手

我还需要看一下 Transformer 的 forward 部分，这样能准确说明它怎么把 action 序列当 token、怎么使用条件和 diffusion timestep。

---

## 38. 助手

你看到的没错：仓库里确实有 **Transformer 版 diffusion denoiser**，也就是 `TransformerForDiffusion`。它和当前实际用的 `UNet diffusion` 的区别不是“是不是 diffusion”，而是：

**两者都是 diffusion policy，区别在于用什么网络来预测噪声/动作。**

当前 RoboTwin 这个 DP 配置用的是：

```text
DiffusionUnetImagePolicy -> ConditionalUnet1D
```

而 `TransformerForDiffusion` 是另一个可选骨干：

```text
TransformerForDiffusion -> Transformer Encoder/Decoder
```

### 共同点

两者输入输出形式很像，都是处理一个 action 序列：

```text
sample:   (B, T, action_dim)
timestep: diffusion step
cond:     observation condition
output:   (B, T, action_dim)
```

也就是说，它们都可以作为 diffusion 里的“去噪网络”：

```text
带噪 action trajectory + diffusion timestep + observation condition
        ↓
预测 noise / denoised action
```

### UNet Diffusion

当前实际使用的是 `ConditionalUnet1D`：

```74:83:policy/DP/diffusion_policy/model/diffusion/conditional_unet1d.py
class ConditionalUnet1D(nn.Module):

    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
```

它的特点是：

- 把 action horizon 看成一维时间序列
- 用 `Conv1D`、downsample、upsample、skip connection 建模
- 类似图像 diffusion 里的 UNet，只是这里是 1D trajectory
- 对短 horizon 的连续动作序列很常见、稳定、训练效率高
- 当前 `robot_dp_14.yaml` / `robot_dp_16.yaml` 就是用这个

它更像：

```text
action sequence -> 1D convolution UNet -> predicted noise
```

### Transformer Diffusion

`TransformerForDiffusion` 则把每个时间步的 action 当成 token：

```46:56:policy/DP/diffusion_policy/model/diffusion/transformer_for_diffusion.py
        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
```

forward 里可以看到，它把 diffusion timestep、condition、action token 送进 Transformer：

```278:318:policy/DP/diffusion_policy/model/diffusion/transformer_for_diffusion.py
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            ...
            x = self.encoder(src=x, mask=self.mask)
            ...
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                ...
            x = self.encoder(x)
            memory = x

            # decoder
            token_embeddings = input_emb
            ...
            x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        x = self.ln_f(x)
        x = self.head(x)
```

它的特点是：

- 用 self-attention 建模 action 时间步之间的关系
- 可以支持 causal attention，像 GPT 那样只看过去
- 也可以用 encoder-decoder，把 observation condition 作为 memory
- 对长序列、复杂时序依赖更灵活
- 但通常参数更多，训练更贵，对数据量和超参更敏感

### 简单对比

| 项目 | UNet Diffusion | Transformer Diffusion |
|---|---|---|
| 主干 | `ConditionalUnet1D` | `TransformerForDiffusion` |
| 时间建模 | 1D 卷积 + 下采样/上采样 | self-attention |
| 序列视角 | 连续轨迹信号 | token 序列 |
| 局部/全局关系 | 卷积偏局部，多尺度扩大感受野 | attention 直接建模全局依赖 |
| 训练成本 | 通常更低、更稳定 | 通常更高、更吃数据 |
| 当前 RoboTwin DP 是否使用 | 是 | 不是当前配置使用 |

所以结论是：

**这个项目代码里同时保留了 Transformer diffusion 的实现，但当前 RoboTwin 的 DP policy 配置实际走的是 UNet diffusion。**

`TransformerForDiffusion` 更像是 diffusion policy 框架里的另一个可替换 denoising backbone，当前没有被 `robot_dp_14.yaml` / `robot_dp_16.yaml` 接入。

---

## 39. 用户

<timestamp>Tuesday, Apr 28, 2026, 12:04 PM (UTC+8)</timestamp>
<user_query>
@policy/DP/diffusion_policy/model 这个文件夹中的内容，作用是什么？
</user_query>

---

## 40. 助手

我会先快速梳理 `policy/DP/diffusion_policy/model` 的目录结构和关键类，再按子目录解释它们在 DP 里的分工。

**Exploring folder content**

I need to provide an answer in Chinese since the user is asking about folder content. First, I'll explore the model folder structure, which will likely require some inspection. I'm considering using tools like Glob or rg to help with this exploration. I don't think I'll need to edit anything here—just gather the information I need to update my commentary effectively. Let's see what I can find!

---

## 41. 助手

`policy/DP/diffusion_policy/model` 这个文件夹主要放的是 **DP 模型相关的底层网络模块和工具模块**。

它不是训练入口，也不是部署入口，而是给 `policy/DP/diffusion_policy/policy/` 和 `workspace/` 调用的“模型组件库”。

整体可以分成几块：

### 1. `model/diffusion`

这是 diffusion policy 的核心网络模块。

主要包括：

- `conditional_unet1d.py`：当前 RoboTwin DP 实际使用的 UNet 去噪网络
- `transformer_for_diffusion.py`：Transformer 版 diffusion 去噪网络，但当前配置没用它
- `ema_model.py`：EMA 模型参数滑动平均
- `mask_generator.py`：生成 action/observation 的 mask，用于 diffusion 训练时决定哪些位置可见、哪些位置要预测
- `conv1d_components.py`：UNet 里用到的 `Conv1dBlock`、下采样、上采样
- `positional_embedding.py`：diffusion timestep 的 sinusoidal embedding

当前实际 DP 路径大概是：

```text
DiffusionUnetImagePolicy
  -> ConditionalUnet1D
  -> Conv1dBlock / Downsample1d / Upsample1d
  -> DDPMScheduler
```

### 2. `model/vision`

这是图像观测编码器相关模块。

主要包括：

- `multi_image_obs_encoder.py`：把多相机 RGB 图像编码成 observation feature
- `model_getter.py`：获取 ResNet、R3M 等视觉 backbone
- `crop_randomizer.py`：图像随机裁剪增强

在当前配置里，视觉部分用的是 ResNet18：

```text
image observation -> MultiImageObsEncoder -> ResNet18 -> obs feature
```

然后这个 feature 会作为 condition 送给 diffusion policy。

### 3. `model/common`

这是通用模型工具。

主要包括：

- `normalizer.py`：归一化器，比如 action/image normalization
- `tensor_util.py`：tensor / numpy / dict 结构的批处理工具
- `rotation_transformer.py`：旋转表示转换，比如 rotvec、matrix、6D rotation
- `lr_scheduler.py`：学习率 scheduler
- `module_attr_mixin.py`：方便从 module 获取 device、dtype 等属性
- `dict_of_tensor_mixin.py`：让一些对象可以像模型一样管理一组 tensor 参数
- `shape_util.py`：推断网络输出 shape、获取 module device

这些不是具体 policy，但训练和推理都经常用。

### 4. `model/bet`

这是 **BET / Behavior Transformer** 相关代码，不是当前 RoboTwin DP 主路径。

里面有：

- `latent_generators/transformer.py`
- `latent_generators/mingpt.py`
- `action_ae/discretizers/k_means.py`
- `libraries/mingpt/`
- `loss_fn.py`

它更偏向另一类行为建模方法：把动作离散成 latent/bin，再用 Transformer 预测 latent。当前你看的 DP 配置主要走 `DiffusionUnetImagePolicy`，不是 BET。

### 简单总结

`model` 文件夹的作用可以理解为：

```text
model/
  common/      通用 tensor、normalizer、rotation、scheduler 工具
  vision/      图像编码器，把 RGB 观测变成 feature
  diffusion/   diffusion policy 的去噪网络、mask、EMA 等核心模块
  bet/         Behavior Transformer / 离散动作建模相关代码
```

当前 RoboTwin 的 DP 实际主要用到的是：

```text
model/vision/multi_image_obs_encoder.py
model/vision/model_getter.py
model/diffusion/conditional_unet1d.py
model/diffusion/mask_generator.py
model/diffusion/ema_model.py
model/common/normalizer.py
```

所以一句话：**这个文件夹是 DP policy 的模型积木库，里面既有当前使用的 UNet diffusion，也有备用/其他算法用的 Transformer 和 BET 组件。**

**Inspecting configuration settings**

I might need to look into the current configuration to understand which model is being used. It could be helpful to read key files, and maybe that's enough to clarify things. I should mention that the model folder isn’t the same as a policy wrapper. It seems like using a folder tree could assist in navigating this. Or perhaps I could apply the "rg" command to target the configuration files. I want to keep everything concise!

---

## 42. 用户

<timestamp>Tuesday, Apr 28, 2026, 12:22 PM (UTC+8)</timestamp>
<user_query>
@policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py 在这个代码中，哪个部分是去噪的过程
</user_query>

---

## 43. 助手

我会定位 `diffusion_unet_image_policy.py` 里训练和推理两条路径：训练时加噪并预测噪声，推理时循环从噪声一步步去噪生成 action。

---

## 44. 助手

在这个文件里，**真正的推理去噪过程**在 `conditional_sample()` 函数里，尤其是这个循环：

```110:118:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample
```

这里的流程是：

1. `trajectory = torch.randn(...)`  
   先从纯随机噪声开始，也就是初始的 noisy action trajectory。

```100:105:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
```

2. `model_output = model(...)`  
   调用 `self.model`，也就是 `ConditionalUnet1D`，预测当前 timestep 下的噪声残差或干净样本。

3. `scheduler.step(...)`  
   根据模型预测结果，把当前 `x_t` 更新成更干净一点的 `x_{t-1}`。

也就是说：

```text
随机噪声 x_T
  -> UNet 预测噪声
  -> scheduler 去掉一部分噪声，得到 x_{T-1}
  -> UNet 再预测噪声
  -> scheduler 再去噪
  ...
  -> 得到最终 action trajectory
```

在 `predict_action()` 中，这一行触发了上面的采样/去噪过程：

```167:174:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )
```

另外，训练时也有“加噪-学去噪”的过程，在 `compute_loss()` 里：

```223:244:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        ...
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        ...
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)
```

这里训练时不是完整循环去噪，而是：

```text
真实 action trajectory
  -> 随机选 timestep
  -> 加对应强度的噪声
  -> UNet 预测噪声
  -> 和真实 noise 做 MSE loss
```

所以总结一下：

- **推理时真正循环去噪**：`conditional_sample()` 里的 `for t in scheduler.timesteps`
- **训练时学习去噪能力**：`compute_loss()` 里的 `add_noise(...)` 和 `self.model(noisy_trajectory, timesteps, ...)`
- **真正执行噪声预测的网络**：`self.model`，也就是 `ConditionalUnet1D`
- **真正根据预测更新 x_t -> x_t-1 的模块**：`self.noise_scheduler.step(...)`

---

## 45. 用户

<timestamp>Tuesday, Apr 28, 2026, 12:30 PM (UTC+8)</timestamp>
<user_query>
时间是不可逆的，这种从T-->T-1的过程，不太好理解，请从正向的角度解释一下推理的过程
</user_query>

---

## 46. 助手

可以从“时间正向执行动作”的角度理解，不要把 `T -> T-1` 想成真实世界时间倒流。

这里的 `T, T-1, ...` 是 **diffusion 的噪声等级编号**，不是机器人真实执行时间。

### 两种时间

在这段代码里其实有两种“时间”：

1. **动作序列时间**  
   例如未来 8 步动作：

```text
a0, a1, a2, a3, a4, a5, a6, a7
```

这是机器人真实要按顺序执行的时间。

2. **diffusion 去噪步数**  
   例如采样 100 步：

```text
noise_level_100 -> noise_level_99 -> ... -> noise_level_0
```

这是模型内部生成动作序列的计算过程，不是机器人执行时间。

所以 `x_t -> x_{t-1}` 的意思是：

```text
更乱的动作序列草稿 -> 更清晰的动作序列草稿
```

不是：

```text
机器人从未来回到过去
```

### 正向理解推理过程

推理时可以这样理解：

**模型先随机写一版未来动作计划，然后反复修改这份计划，直到它变成合理动作。**

比如要生成未来 8 步动作，最开始是随机噪声：

```text
step 0: 随机动作草稿
[a?, a?, a?, a?, a?, a?, a?, a?]
```

然后每一轮，UNet 看当前草稿、当前观测图像、diffusion step，判断“这份草稿里面哪些部分像噪声”，scheduler 根据这个判断把草稿修正一点：

```text
step 1: 稍微像动作
[a~, a~, a~, a~, a~, a~, a~, a~]

step 2: 更像动作
[a1?, a2?, a3?, a4?, a5?, a6?, a7?, a8?]

...

final: 清晰动作序列
[a0, a1, a2, a3, a4, a5, a6, a7]
```

对应代码是：

```110:118:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample
```

从正向角度说，这个循环就是：

```text
第 1 次修正计划
第 2 次修正计划
第 3 次修正计划
...
第 100 次修正计划
得到最终未来动作序列
```

`x_t -> x_{t-1}` 只是数学上表示“噪声等级降低一级”。

### 最后机器人怎么执行？

生成完以后，代码取其中一段动作：

```180:183:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
```

所以真实执行仍然是正向的：

```text
当前观测
  -> 生成未来动作计划 [a0, a1, a2, ...]
  -> 执行 a0
  -> 执行 a1
  -> 执行 a2
```

核心理解：

**diffusion 的 `T -> T-1` 是“噪声程度从高到低”，不是“物理时间从未来到过去”。推理过程正向看，就是不断把一份随机动作草稿修改成清晰、可执行的未来动作计划。**

---

## 47. 用户

<attached_files>

<code_selection path="/home/lzy/code/RoboTwin/policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py" lines="68-68">
L68:            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
</code_selection>

<code_selection path="/home/lzy/code/RoboTwin/policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py" lines="164-164">
L164:            cond_data[:, :To, Da:] = nobs_features
</code_selection>

</attached_files>
<timestamp>Tuesday, Apr 28, 2026, 1:02 PM (UTC+8)</timestamp>
<user_query>
@policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py:68 这里obs_as_global_cond是什么意思，为什么为true时obs_dim=0 @policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py:164 为什么obs_as_global_cond为False，cond_data才有值？
</user_query>

---

## 48. 助手

我会结合初始化、`predict_action` 和训练里的两条分支解释：`obs_as_global_cond=True` 是把观测作为全局条件传给 UNet，而 `False` 是把观测特征拼进 trajectory 里做 inpainting。

---

## 49. 助手

`obs_as_global_cond` 的意思是：**观测信息 obs 要不要作为 UNet 的“全局条件”输入。**

这里有两种把 obs 喂给 diffusion model 的方式。

### 情况 1：`obs_as_global_cond=True`

这是当前配置用的方式。

代码里会这样设置：

```45:55:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
```

意思是：

```text
UNet 主输入 sample 里只放 action
obs 特征不拼进 sample
obs 特征作为 global_cond 额外传给 UNet
```

所以 `input_dim = action_dim`。

推理时也是这样：

```147:155:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
```

这里 `cond_data` 不是“没有条件”，而是：**条件不放在 cond_data 里，而是放在 `global_cond` 里。**

所以你问为什么 `True` 时：

```python
obs_dim = 0
```

原因是 `LowdimMaskGenerator` 只管理 `trajectory` 里的维度。

当 `obs_as_global_cond=True` 时，trajectory 只有 action：

```text
trajectory shape = (B, T, action_dim)
```

obs 不在 trajectory 里，所以：

```python
obs_dim = 0
```

此时 `condition_mask` 也全是 False，因为没有需要固定的 obs 维度：

```text
trajectory: [action]
mask:       [False]
```

obs 是通过这里进入 UNet 的：

```237:238:policy/DP/diffusion_policy/model/diffusion/conditional_unet1d.py
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
```

也就是把 diffusion timestep embedding 和 obs feature 拼起来，作为每个 ResBlock 的条件。

### 情况 2：`obs_as_global_cond=False`

这时 obs 不作为全局条件，而是直接拼进 trajectory：

```text
trajectory = [action, obs_feature]
```

所以推理时才会有这两行：

```162:165:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True
```

含义是：

```text
前 To 个时间步的 obs_feature 是已知条件
把它填进 cond_data
并用 cond_mask=True 标记这些位置不能被 diffusion 改掉
```

然后在采样循环里，每一步都会强制把这些观测特征写回去：

```110:112:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
```

这叫 **inpainting conditioning**：  
一部分 trajectory 是已知的 obs，另一部分 action 是要生成的。

### 为什么 `False` 时 `cond_data` 才有 obs 值？

因为两种条件路径不同：

```text
obs_as_global_cond=True:
  obs -> obs_encoder -> global_cond
  trajectory 只包含 action
  cond_data 只是 action 形状的空容器

obs_as_global_cond=False:
  obs -> obs_encoder -> 拼进 trajectory 后半部分
  trajectory 包含 action + obs_feature
  cond_data 需要保存已知 obs_feature
  cond_mask 标记 obs_feature 固定不变
```

所以不是 `True` 时没有用 obs，而是用法变了：

**True：obs 作为全局条件调制 UNet。**  
**False：obs 作为 trajectory 的一部分，通过 mask 固定住，让模型补全 action。**

当前配置里是：

```text
obs_as_global_cond: True
```

所以 RoboTwin 这里实际走的是第一种：**图像观测编码成全局条件，UNet 只对 action trajectory 做 diffusion 去噪。**

---

## 50. 用户

<timestamp>Tuesday, Apr 28, 2026, 1:03 PM (UTC+8)</timestamp>
<user_query>
cond_data和global_cond有什么区别
</user_query>

---

## 51. 助手

`cond_data` 和 `global_cond` 都是“条件”，但它们进入模型的位置完全不同。

`global_cond` 是 **额外传给 UNet 的全局条件向量**。  
它不属于被 diffusion 去噪的 trajectory 本体，而是像“提示信息”一样调制 UNet。

在当前 `obs_as_global_cond=True` 的路径里：

```python
global_cond = nobs_features.reshape(B, -1)
cond_data = torch.zeros(size=(B, T, Da), ...)
cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
```

含义是：

```text
global_cond: 当前观测图像编码后的特征
cond_data:   action trajectory 的占位数据，里面没有 obs
```

`global_cond` 会进入 `ConditionalUnet1D`：

```237:238:policy/DP/diffusion_policy/model/diffusion/conditional_unet1d.py
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
```

也就是：

```text
diffusion timestep embedding + obs feature
        ↓
调制 UNet 每个残差块
```

而 `cond_data` 是 **trajectory 里已经确定、不能被 diffusion 改掉的部分**。  
它要配合 `cond_mask` 使用：

```110:112:policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
```

也就是说：

```text
cond_mask=True 的位置：
  每次去噪前都强制写回 cond_data

cond_mask=False 的位置：
  由 diffusion 生成
```

### 一个直观区别

`global_cond` 像是给模型看的“背景信息”：

```text
当前图像是什么、机器人看到什么、任务环境是什么
```

`cond_data` 像是 trajectory 中“已经钉死的已知值”：

```text
这些时间步/这些维度已经知道了，不允许模型改
```

### 当前代码里的两种模式

#### `obs_as_global_cond=True`

```text
trajectory = [action]
global_cond = obs_feature
cond_data = 全 0
cond_mask = 全 False
```

obs 不在 trajectory 里面，而是作为全局条件输入 UNet。

#### `obs_as_global_cond=False`

```text
trajectory = [action, obs_feature]
global_cond = None
cond_data[:, :To, Da:] = obs_feature
cond_mask[:, :To, Da:] = True
```

obs 被拼进 trajectory，前 `To` 步 obs 是已知的，所以用 `cond_data + cond_mask` 固定住。

一句话总结：

**`global_cond` 是模型外部的条件提示；`cond_data` 是 diffusion 采样对象内部被 mask 固定的已知部分。**

---

## 52. 用户

<timestamp>Sunday, Jun 7, 2026, 2:45 PM (UTC+8)</timestamp>
<user_query>
把上面的这段对话, 导出md文件到 talk/文件夹
</user_query>

---

## 53. 助手

我会把当前这段关于 DP 代码理解的问答整理成一个 Markdown 文档，并放到仓库的 `talk/` 目录下。先确认目录是否存在。

---

## 54. 助手

`talk/` 目录已经存在。我现在生成一个整理版 Markdown，包含我们刚才围绕 `policy/DP` 的主要问题和结论。

**Creating markdown notes**

I’m planning to create a markdown file that’s both concise and detailed, perhaps naming it DP_conversation_notes.md or dp_code_discussion.md. First, I want to check if a file with that name already exists. I'll look for any markdown files in the talk directory to ensure there's no overlap. This way, I can avoid confusion in my documentation process!

---

## 55. 助手

我会命名为 `talk/dp_code_discussion.md`，内容按主题整理，方便后续回看。

**Planning file adjustments**

I need to apply a patch and add a file. It would be helpful to include the whole conversation, so I’ll probably need all Q&A from the dp_model workspace related to various technical aspects. It could be useful to mention test changes too, along with the file path. I want to keep everything concise and not make it too massive. I'll remember to use ApplyPatch in this process.

---

## 56. 助手

Markdown 已经生成。我再简单读一下文件开头，确认路径和内容写入正常。

---

## 57. 助手

已导出到：

`talk/dp_code_discussion.md`

我把上面关于 `policy/DP`、UNet diffusion、Transformer diffusion、去噪流程、`obs_as_global_cond`、`cond_data/global_cond` 等内容整理成了 Markdown 笔记。

---

## 58. 用户

<timestamp>Sunday, Jun 7, 2026, 2:46 PM (UTC+8)</timestamp>
<user_query>
这个是整个session的所有对话吗？
</user_query>

---

## 59. 助手

不是。现在这个 `talk/dp_code_discussion.md` 是我把**上面这段围绕 `policy/DP` 的主要技术讨论**整理成的笔记版，不是逐字逐句的整个 session 全量对话。

它包含了主要主题和结论，但没有包含所有原始问答、终端输出、中间操作提示、测试改动细节等。

如果你想要“整个 session 的完整对话记录”，我可以再导出一个更完整的版本到 `talk/`，比如：

`talk/full_session_dialogue.md`

内容按“用户 / 助手”逐轮记录。

---

## 60. 用户

<timestamp>Sunday, Jun 7, 2026, 2:56 PM (UTC+8)</timestamp>
<user_query>
再直接导出当前会话原版本的完整对话记录，不需要总结的版本
</user_query>

---

## 61. 助手

我会导出一个“不总结、不改写”的完整对话版 Markdown，尽量按当前会话的用户/助手原始问答顺序保留。先确认是否有现成会话记录可用，避免漏掉早期内容。

---

## 62. 助手

我看到 `talk/` 里已经有一个 `session_raw_2026-06-07_dp_train_eval.md`，我先看一下它是否就是原始逐轮对话；如果不是，我会重新生成/覆盖一个完整版本。

---

## 63. 用户

<timestamp>Sunday, Jun 7, 2026, 3:07 PM (UTC+8)</timestamp>
<user_query>
再直接导出当前会话原版本的完整对话记录，不需要总结的版本
</user_query>

---
