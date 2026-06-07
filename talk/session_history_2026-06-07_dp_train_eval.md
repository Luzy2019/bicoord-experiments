# 2026-06-07 DP 训练与评估会话记录

本文整理本次会话中围绕 RoboTwin `policy/DP` 的继续训练、评估 checkpoint、episode 数量、资源缺失和下载脚本问题的完整讨论记录。

## 1. 基于已有 400 checkpoint 继续训练

用户最初的问题：

```bash
bash train.sh scan_object aloha-agilex_clean_50 50 0 14 0
```

希望在 `policy/DP/checkpoints` 下已经训练好的 `400.ckpt` 基础上继续训练，并询问如何修改脚本。

分析结果：

- `policy/DP/train.sh` 会调用 `train.py`。
- `policy/DP/diffusion_policy/workspace/robotworkspace.py` 中已有 `cfg.training.resume` 逻辑。
- 但原始代码只会找当前 Hydra 输出目录下的 `checkpoints/latest.ckpt`。
- 实际保存 checkpoint 的代码是：

```python
self.save_checkpoint(f"checkpoints/{save_name}-{seed}/{self.epoch + 1}.ckpt")
```

也就是保存为类似：

```text
checkpoints/scan_object-aloha-agilex_clean_50-50-0/400.ckpt
```

因此原始逻辑不会自动找到 `400.ckpt`。

随后根据用户要求，代码已修改为：继续使用相同命令时自动查找同一任务、同一 seed 的最新 `.ckpt`。

继续训练命令保持不变：

```bash
bash train.sh scan_object aloha-agilex_clean_50 50 0 14 0
```

实际修改文件：

```text
policy/DP/diffusion_policy/workspace/robotworkspace.py
```

主要改动：

- 新增 `get_auto_resume_checkpoint_path()`。
- 优先查找 Hydra 输出目录的 `latest.ckpt`。
- 如果没有 `latest.ckpt`，自动查找：

```text
checkpoints/<zarr文件名>-<seed>/*.ckpt
policy/DP/checkpoints/<zarr文件名>-<seed>/*.ckpt
```

- 如果找到 `400.ckpt`、`800.ckpt` 等数字命名 checkpoint，会选择最新的一个。
- 恢复后将 `self.epoch` 至少设置到 checkpoint 文件名对应的 epoch。
- 训练循环从 `self.epoch` 开始，而不是重新从 0 开始。

验证：

```bash
python3 -m py_compile policy/DP/diffusion_policy/workspace/robotworkspace.py
```

检查通过。

## 2. 评估命令使用哪个 checkpoint

用户执行：

```bash
bash eval.sh scan_object demo_clean aloha-agilex_clean_50 50 0 0
```

询问这个脚本使用的是哪个 checkpoint。

分析 `policy/DP/eval.sh` 参数：

```bash
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
```

对应为：

```text
task_name=scan_object
task_config=demo_clean
ckpt_setting=aloha-agilex_clean_50
expert_data_num=50
seed=0
gpu_id=0
```

`policy/DP/deploy_policy.yml` 中：

```yaml
checkpoint_num: 600
```

`policy/DP/deploy_policy.py` 中 checkpoint 路径拼接逻辑为：

```python
ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
```

所以该评估命令使用：

```text
./policy/DP/checkpoints/scan_object-aloha-agilex_clean_50-50-0/600.ckpt
```

如果要评估 `400.ckpt`，需要把 `policy/DP/deploy_policy.yml` 中的：

```yaml
checkpoint_num: 600
```

改为：

```yaml
checkpoint_num: 400
```

## 3. 一共测试多少个 episode

用户询问评估脚本一共测试多少个 episode。

分析 `script/eval_policy.py`：

```python
test_num = 100
```

随后传入：

```python
eval_policy(..., test_num=test_num, ...)
```

结论：

```text
默认一共测试 100 个 episode。
```

## 4. 评估时报缺少 Objaverse 资源

用户运行评估后遇到错误：

```text
FileNotFoundError: [Errno 2] No such file or directory: 'assets/objects/objaverse/list.json'
```

错误链路：

```text
script/eval_policy.py
-> envs
-> envs/utils/rand_create_cluttered_actor.py
-> get_all_cluttered_objects()
-> assets/objects/objaverse/list.json
```

结论：

- 这不是 checkpoint 问题。
- 也不是 `eval.sh` 参数问题。
- 是 RoboTwin 所需的 assets 没有下载或没有放到正确目录。

需要补齐：

```text
assets/objects/objaverse/list.json
```

以及对应 object assets。

## 5. 下载 assets 时代理错误

用户执行：

```bash
bash script/_download_assets.sh
```

遇到错误：

```text
ProxyError: Unable to connect to proxy
HTTPSConnection(host='127.0.0.1', port=9674): Failed to establish a new connection
```

分析结果：

- `script/_download_assets.sh` 原来写死了代理：

```bash
export http_proxy="http://127.0.0.1:9674"
export https_proxy="https://127.0.0.1:9674"
export all_proxy="socks5h://127.0.0.1:9674"
```

- 但本机 `127.0.0.1:9674` 没有代理服务在监听。
- 因此 HuggingFace 下载失败。
- 后面的 unzip 错误只是连带结果，因为 zip 文件没有下载下来：

```text
unzip: cannot find or open background_texture.zip
unzip: cannot find or open embodiments.zip
unzip: cannot find or open objects.zip
```

## 6. 自动处理下载脚本

用户要求：

```text
你可以自己执行，直到能够成功执行eval脚本为止
```

执行前检查到：

- `script/_download_assets.sh` 写死了坏代理。
- 工具环境中没有 `python` 命令，只有 `python3`。
- 但 RoboTwin conda 环境里有可用 Python：

```text
/home/lzy/anaconda3/envs/RoboTwin/bin/python
```

因此修改了：

```text
script/_download_assets.sh
```

主要改动：

```bash
set -e

PYTHON_BIN=${PYTHON_BIN:-python3}

cd assets
${PYTHON_BIN} _download.py
```

也就是：

- 去掉写死的 `127.0.0.1:9674` 代理。
- 增加 `set -e`，下载或解压失败时立即退出。
- 默认使用 `python3`。
- 允许通过环境变量 `PYTHON_BIN` 指定 Python。

之后尝试用系统 `python3` 下载，报错：

```text
ModuleNotFoundError: No module named 'huggingface_hub'
```

于是改用 RoboTwin conda 环境：

```bash
PYTHON_BIN="/home/lzy/anaconda3/envs/RoboTwin/bin/python" bash script/_download_assets.sh
```

下载开始后，用户说明：

```text
不用了，已经下载完了
```

因此停止继续处理下载和评估流程。

最后提醒用户：

- `script/_download_assets.sh` 已经被修改。
- 后续再运行下载脚本时，不会再强制走坏掉的 `127.0.0.1:9674` 代理。

## 7. 本次会话涉及的文件

读取或分析过的主要文件：

```text
policy/DP/train.sh
policy/DP/train.py
policy/DP/diffusion_policy/workspace/robotworkspace.py
policy/DP/diffusion_policy/workspace/base_workspace.py
policy/DP/diffusion_policy/config/robot_dp_14.yaml
policy/DP/eval.sh
policy/DP/deploy_policy.yml
policy/DP/deploy_policy.py
policy/DP/dp_model.py
script/eval_policy.py
script/_download_assets.sh
assets/_download.py
script/update_embodiment_config_path.py
```

实际修改过的文件：

```text
policy/DP/diffusion_policy/workspace/robotworkspace.py
script/_download_assets.sh
talk/session_history_2026-06-07_dp_train_eval.md
```

## 8. 当前可用命令总结

继续训练：

```bash
cd /home/lzy/code/RoboTwin/policy/DP
bash train.sh scan_object aloha-agilex_clean_50 50 0 14 0
```

评估：

```bash
cd /home/lzy/code/RoboTwin/policy/DP
bash eval.sh scan_object demo_clean aloha-agilex_clean_50 50 0 0
```

评估默认 checkpoint：

```text
policy/DP/checkpoints/scan_object-aloha-agilex_clean_50-50-0/600.ckpt
```

评估默认 episode 数：

```text
100
```
