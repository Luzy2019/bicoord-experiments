import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def parse_args():
    parser = argparse.ArgumentParser(description="Compare executed factorized DP actions with zarr training actions.")
    parser.add_argument(
        "--debug-file",
        type=Path,
        default=Path("../../debug_factorized/factorized_debug.npz"),
        help="Path to factorized_debug.npz.",
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("data/handover_block_with_bowls-demo_clean-50.zarr"),
        help="Path to training zarr.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("../../debug_factorized/train_compare"))
    parser.add_argument("--episode-len", type=int, default=350)
    parser.add_argument("--action-start", type=int, default=2)
    parser.add_argument("--action-steps", type=int, default=6)
    parser.add_argument("--left-dim", type=int, default=7)
    return parser.parse_args()


def mse(a, b, axis=None):
    return np.mean((a - b) ** 2, axis=axis)


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) == target_len:
        return seq.copy()
    src_x = np.linspace(0.0, 1.0, len(seq))
    dst_x = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for dim in range(seq.shape[1]):
        out[:, dim] = np.interp(dst_x, src_x, seq[:, dim])
    return out


def load_executed_action(debug_file: Path, action_start: int, action_steps: int, episode_len: int):
    data = np.load(debug_file)
    final_action = data["final_action"][:, 0]  # [policy_call, horizon, action_dim]
    executed = final_action[:, action_start:action_start + action_steps, :].reshape(-1, final_action.shape[-1])
    return executed[:episode_len]


def load_train_episodes(zarr_path: Path):
    root = zarr.open(str(zarr_path), mode="r")
    actions = root["data"]["action"][:]
    episode_ends = root["meta"]["episode_ends"][:]

    episodes = []
    start = 0
    for end in episode_ends:
        episodes.append(actions[start:int(end)])
        start = int(end)
    return episodes


def phase_stats(values: np.ndarray):
    n = len(values)
    phases = [
        ("early", 0, n // 3),
        ("middle", n // 3, 2 * n // 3),
        ("late", 2 * n // 3, n),
    ]
    return [(name, float(values[s:e].mean()), float(values[s:e].max())) for name, s, e in phases]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    executed = load_executed_action(args.debug_file, args.action_start, args.action_steps, args.episode_len)
    train_episodes = load_train_episodes(args.zarr_path)
    train_resampled = np.stack([resample_sequence(ep, len(executed)) for ep in train_episodes], axis=0)
    np.save(args.output_dir / "executed_action_350.npy", executed)
    np.savetxt(args.output_dir / "executed_action_350.csv", executed, delimiter=",")

    left_slice = slice(0, args.left_dim)
    right_slice = slice(args.left_dim, executed.shape[-1])

    per_episode_mse = mse(train_resampled, executed[None, :, :], axis=(1, 2))
    best_ep = int(np.argmin(per_episode_mse))
    best_train = train_resampled[best_ep]
    train_mean = train_resampled.mean(axis=0)
    np.save(args.output_dir / "closest_train_action_resampled_350.npy", best_train)
    np.savetxt(args.output_dir / "closest_train_action_resampled_350.csv", best_train, delimiter=",")

    left_err = mse(executed[:, left_slice], best_train[:, left_slice], axis=1)
    right_err = mse(executed[:, right_slice], best_train[:, right_slice], axis=1)
    both_err = 0.5 * (left_err + right_err)

    left_err_mean = mse(executed[:, left_slice], train_mean[:, left_slice], axis=1)
    right_err_mean = mse(executed[:, right_slice], train_mean[:, right_slice], axis=1)

    steps = np.arange(len(executed))
    top_left = np.argsort(left_err)[-10:][::-1]
    top_right = np.argsort(right_err)[-10:][::-1]
    top_both = np.argsort(both_err)[-10:][::-1]

    plt.figure(figsize=(12, 5))
    plt.plot(steps, left_err, label="left vs closest train episode")
    plt.plot(steps, right_err, label="right vs closest train episode")
    plt.plot(steps, both_err, label="mean(left,right)", alpha=0.8)
    plt.scatter(top_both[:5], both_err[top_both[:5]], color="red", label="top mean peaks")
    for idx in top_both[:5]:
        plt.annotate(str(idx), (idx, both_err[idx]), textcoords="offset points", xytext=(0, 6), ha="center")
    plt.xlabel("executed action step")
    plt.ylabel("MSE over arm dims")
    plt.title(f"Executed action vs closest training episode {best_ep}")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig(args.output_dir / "01_executed_vs_closest_train_mse.png")

    plt.figure(figsize=(12, 5))
    plt.plot(steps, left_err_mean, label="left vs train mean")
    plt.plot(steps, right_err_mean, label="right vs train mean")
    plt.xlabel("executed action step")
    plt.ylabel("MSE over arm dims")
    plt.title("Executed action vs mean training trajectory")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig(args.output_dir / "02_executed_vs_train_mean_mse.png")

    def plot_arm(prefix, arm_slice, dim_count):
        fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        axes = axes.reshape(-1)
        for dim in range(dim_count):
            ax = axes[dim]
            global_dim = arm_slice.start + dim
            ax.plot(steps, executed[:, global_dim], label="executed")
            ax.plot(steps, best_train[:, global_dim], label=f"closest train ep {best_ep}")
            ax.plot(steps, train_mean[:, global_dim], label="train mean", alpha=0.7)
            ax.set_title(f"{prefix} dim {dim}")
            ax.grid(alpha=0.3)
        axes[-1].axis("off")
        axes[0].legend()
        fig.suptitle(f"{prefix} action trajectories")
        savefig(args.output_dir / f"03_{prefix}_action_trajectories.png")

    plot_arm("left", left_slice, args.left_dim)
    plot_arm("right", right_slice, executed.shape[-1] - args.left_dim)

    dim_left_mse = mse(executed[:, left_slice], best_train[:, left_slice], axis=0)
    dim_right_mse = mse(executed[:, right_slice], best_train[:, right_slice], axis=0)
    plt.figure(figsize=(9, 4))
    labels = [f"L{i}" for i in range(len(dim_left_mse))] + [f"R{i}" for i in range(len(dim_right_mse))]
    vals = np.concatenate([dim_left_mse, dim_right_mse])
    plt.bar(labels, vals)
    plt.ylabel("MSE over executed steps")
    plt.title("Per-dimension executed vs closest train episode MSE")
    plt.grid(axis="y", alpha=0.3)
    savefig(args.output_dir / "04_per_dim_mse.png")

    summary = []
    summary.append(f"debug_file: {args.debug_file}")
    summary.append(f"zarr_path: {args.zarr_path}")
    summary.append(f"executed_steps: {len(executed)}")
    summary.append(f"train_episodes: {len(train_episodes)}")
    summary.append(f"closest_train_episode: {best_ep}")
    summary.append(f"closest_train_episode_raw_len: {len(train_episodes[best_ep])}")
    summary.append(f"closest_full_action_mse: {per_episode_mse[best_ep]:.8f}")
    summary.append("")
    summary.append(f"left mse mean/max vs closest: {left_err.mean():.8f} / {left_err.max():.8f}")
    summary.append(f"right mse mean/max vs closest: {right_err.mean():.8f} / {right_err.max():.8f}")
    summary.append(f"left phase mean/max early-middle-late: {phase_stats(left_err)}")
    summary.append(f"right phase mean/max early-middle-late: {phase_stats(right_err)}")
    summary.append("")
    summary.append("top left peaks: " + ", ".join([f"step {idx} mse {left_err[idx]:.8f}" for idx in top_left]))
    summary.append("top right peaks: " + ", ".join([f"step {idx} mse {right_err[idx]:.8f}" for idx in top_right]))
    summary.append("top mean peaks: " + ", ".join([f"step {idx} mse {both_err[idx]:.8f}" for idx in top_both]))
    summary.append("")
    summary.append(f"left per-dim mse: {dim_left_mse.tolist()}")
    summary.append(f"right per-dim mse: {dim_right_mse.tolist()}")
    summary.append(f"plots saved to: {args.output_dir}")
    (args.output_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
