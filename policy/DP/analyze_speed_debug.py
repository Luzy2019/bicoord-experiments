import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze learned speed modulation debug npz.")
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--action-start", type=int, default=2)
    parser.add_argument("--action-steps", type=int, default=6)
    parser.add_argument("--fps", type=float, default=60.0)
    return parser.parse_args()


def executed_slice(array: np.ndarray, start: int, steps: int) -> np.ndarray:
    return array[:, :, start : start + steps]


def flatten_episode(alpha_exec: np.ndarray) -> np.ndarray:
    """Convert [policy_call, batch, action_step] to one episode step series."""
    if alpha_exec.ndim == 4:
        alpha_exec = alpha_exec[:, 0]
    if alpha_exec.ndim == 3:
        alpha_exec = alpha_exec[:, 0]
    return alpha_exec.reshape(-1)


def summarize_alpha(name: str, alpha: np.ndarray):
    flat = alpha.reshape(-1)
    print(f"{name}:")
    print(f"  mean={flat.mean():.4f}, min={flat.min():.4f}, max={flat.max():.4f}")
    print(f"  pct(alpha>1.0)={(flat > 1.0).mean() * 100:.2f}%")
    print(f"  pct(alpha>1.2)={(flat > 1.2).mean() * 100:.2f}%")
    print(f"  pct(alpha<0.8)={(flat < 0.8).mean() * 100:.2f}%")


def plot_alpha(output_dir: Path, name: str, alpha_exec: np.ndarray, fps: float):
    series = flatten_episode(alpha_exec)
    steps = np.arange(series.shape[0])
    seconds = steps / fps
    plt.figure(figsize=(12, 4))
    plt.plot(steps, series, linewidth=1.5)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title(f"{name} executed alpha over episode steps")
    plt.xlabel("executed step")
    plt.ylabel("alpha")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_alpha_over_steps.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(seconds, series, linewidth=1.5)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title(f"{name} executed alpha over episode time")
    plt.xlabel("time (s)")
    plt.ylabel("alpha")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_alpha_over_time.png", dpi=160)
    plt.close()


def plot_bimanual_alpha(output_dir: Path, left_exec: np.ndarray, right_exec: np.ndarray, fps: float):
    left = flatten_episode(left_exec)
    right = flatten_episode(right_exec)
    steps = np.arange(min(left.shape[0], right.shape[0]))
    left = left[: steps.shape[0]]
    right = right[: steps.shape[0]]
    seconds = steps / fps

    plt.figure(figsize=(14, 5))
    plt.plot(steps, left, label="left alpha", linewidth=1.5)
    plt.plot(steps, right, label="right alpha", linewidth=1.5)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.fill_between(steps, left, right, where=right > left, alpha=0.2, label="right faster")
    plt.fill_between(steps, left, right, where=left > right, alpha=0.2, label="left faster")
    plt.title("Bimanual executed alpha over episode steps")
    plt.xlabel("executed step")
    plt.ylabel("alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bimanual_alpha_over_steps.png", dpi=160)
    plt.close()

    diff = right - left
    plt.figure(figsize=(14, 4))
    plt.plot(steps, diff, linewidth=1.5)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Right alpha - left alpha over episode steps")
    plt.xlabel("executed step")
    plt.ylabel("right - left alpha")
    plt.tight_layout()
    plt.savefig(output_dir / "right_minus_left_alpha_over_steps.png", dpi=160)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(seconds, left, label="left alpha", linewidth=1.5)
    plt.plot(seconds, right, label="right alpha", linewidth=1.5)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title("Bimanual executed alpha over episode time")
    plt.xlabel("time (s)")
    plt.ylabel("alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bimanual_alpha_over_time.png", dpi=160)
    plt.close()


def save_episode_csv(output_dir: Path, **series):
    names = list(series.keys())
    length = min(np.asarray(series[name]).shape[0] for name in names)
    table = np.column_stack([np.asarray(series[name])[:length] for name in names])
    header = ",".join(names)
    np.savetxt(output_dir / "episode_alpha.csv", table, delimiter=",", header=header, comments="")


def main():
    args = parse_args()
    data = np.load(args.npz_path)
    output_dir = args.output_dir or args.npz_path.parent / "speed_debug_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {args.npz_path}")
    print("Available keys:", ", ".join(sorted(data.files)))

    if "left_speed_alpha" in data and "right_speed_alpha" in data:
        left = executed_slice(data["left_speed_alpha"], args.action_start, args.action_steps)
        right = executed_slice(data["right_speed_alpha"], args.action_start, args.action_steps)
        summarize_alpha("left_speed_alpha", left)
        summarize_alpha("right_speed_alpha", right)
        left_series = flatten_episode(left)
        right_series = flatten_episode(right)
        diff = right_series - left_series
        print("right-left alpha:")
        print(f"  mean={diff.mean():.4f}, pct(right>left)={(diff > 0).mean() * 100:.2f}%")
        plot_alpha(output_dir, "left", left, fps=args.fps)
        plot_alpha(output_dir, "right", right, fps=args.fps)
        plot_bimanual_alpha(output_dir, left, right, fps=args.fps)
        steps = np.arange(min(left_series.shape[0], right_series.shape[0]))
        save_episode_csv(
            output_dir,
            episode_step=steps,
            time_sec=steps / args.fps,
            left_alpha=left_series[: steps.shape[0]],
            right_alpha=right_series[: steps.shape[0]],
            right_minus_left=diff[: steps.shape[0]],
        )
    elif "speed_alpha" in data:
        alpha = executed_slice(data["speed_alpha"], args.action_start, args.action_steps)
        summarize_alpha("speed_alpha", alpha)
        series = flatten_episode(alpha)
        plot_alpha(output_dir, "joint", alpha, fps=args.fps)
        steps = np.arange(series.shape[0])
        save_episode_csv(
            output_dir,
            episode_step=steps,
            time_sec=steps / args.fps,
            alpha=series,
        )
    else:
        raise KeyError("No speed alpha found. Re-run eval with a checkpoint after the debug logging patch.")

    if "action_pred_raw" in data and "final_action" in data:
        raw = executed_slice(data["action_pred_raw"], args.action_start, args.action_steps)
        final = executed_slice(data["final_action"], args.action_start, args.action_steps)
        mse = ((final - raw) ** 2).mean(axis=(-1, -2, -3))
        print(f"warp action MSE mean={mse.mean():.6f}, max={mse.max():.6f}")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
