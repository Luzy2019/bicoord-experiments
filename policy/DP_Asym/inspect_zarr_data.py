import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset


zarr_path = ROOT / "data/handover_block_with_bowls-demo_clean-50.zarr"

print("==== Raw zarr ====")
root = zarr.open(str(zarr_path), mode="r")

print("groups:", list(root.keys()))
print("data keys:", list(root["data"].keys()))
print("meta keys:", list(root["meta"].keys()))

episode_ends = root["meta"]["episode_ends"][:]
print("num episodes:", len(episode_ends))
print("episode_ends:", episode_ends[:10], "...")

for key in root["data"].keys():
    arr = root["data"][key]
    print(key, arr.shape, arr.dtype)

action = root["data"]["action"][:]
state = root["data"]["state"][:]

print("\n==== Action stats ====")
print("action shape:", action.shape)
print("action min:", action.min(axis=0))
print("action max:", action.max(axis=0))
print("action mean:", action.mean(axis=0))
print("action std:", action.std(axis=0))

left_action = action[:, :7]
right_action = action[:, 7:14]

print("\nleft action std mean:", left_action.std(axis=0).mean())
print("right action std mean:", right_action.std(axis=0).mean())

left_vel = np.diff(left_action, axis=0)
right_vel = np.diff(right_action, axis=0)

print("left velocity abs mean:", np.abs(left_vel).mean())
print("right velocity abs mean:", np.abs(right_vel).mean())
print("left velocity abs max:", np.abs(left_vel).max())
print("right velocity abs max:", np.abs(right_vel).max())

print("\n==== Dataset loader ====")
dataset = RobotImageDataset(
    zarr_path=str(zarr_path),
    horizon=8,
    pad_before=2,
    pad_after=7,
    seed=42,
    val_ratio=0.02,
    batch_size=64,
    max_train_episodes=None,
)

print("dataset len:", len(dataset))
print("train episodes:", dataset.train_mask.sum())
print("val episodes:", (~dataset.train_mask).sum())

sample = dataset[0]
for key, value in sample.items():
    print(key, value.shape, value.dtype)

print("\nfirst sample action:")
print(sample["action"])

print("\nfirst sample state:")
print(sample["state"])

print("\n==== Per-episode velocity check ====")

prev_end = 0
bad_eps = []
left_max_list = []
right_max_list = []

for ep_id, end in enumerate(episode_ends):
    ep_action = action[prev_end:end]
    prev_end = end

    if len(ep_action) < 2:
        continue

    left_vel = np.diff(ep_action[:, :7], axis=0)
    right_vel = np.diff(ep_action[:, 7:14], axis=0)

    left_max = np.abs(left_vel).max()
    right_max = np.abs(right_vel).max()

    left_max_list.append(left_max)
    right_max_list.append(right_max)

    if left_max > 0.5 or right_max > 0.5:
        bad_eps.append((ep_id, len(ep_action), float(left_max), float(right_max)))

print("left per-episode max mean:", np.mean(left_max_list))
print("right per-episode max mean:", np.mean(right_max_list))
print("left per-episode max global:", np.max(left_max_list))
print("right per-episode max global:", np.max(right_max_list))

print("episodes with large jump > 0.5:")
for item in bad_eps[:20]:
    print(item)
print("num bad episodes:", len(bad_eps))