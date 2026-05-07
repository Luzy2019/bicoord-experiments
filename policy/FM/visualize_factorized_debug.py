import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize factorized bimanual DP debug npz.")
    parser.add_argument(
        "debug_file",
        type=Path,
        help="Path to factorized_debug.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_factorized/plots"),
        help="Directory to save plots and summary.",
    )
    parser.add_argument("--policy-call", type=int, default=0, help="Policy call index for trajectory plots.")
    parser.add_argument("--diffusion-index", type=int, default=-1, help="Diffusion index for trajectory plots.")
    parser.add_argument("--action-start", type=int, default=2, help="Start index of executed action chunk.")
    parser.add_argument("--action-steps", type=int, default=6, help="Number of executed actions per policy call.")
    return parser.parse_args()


def mse(a, b, axis=None):
    return np.mean((a - b) ** 2, axis=axis)


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.debug_file)

    timesteps = data["timesteps"]
    gates = data["factorized_gates"][..., 0, :]  # [call, diffusion, 2]
    left_marginal = data["left_marginal"][..., 0, :, :]  # [call, diffusion, horizon, dim]
    right_marginal = data["right_marginal"][..., 0, :, :]
    left_cond = data["left_cond"][..., 0, :, :]
    right_cond = data["right_cond"][..., 0, :, :]
    left_pred = data["left_pred"][..., 0, :, :]
    right_pred = data["right_pred"][..., 0, :, :]

    call_axis = tuple(range(gates.ndim - 1))
    w_mean = gates[..., 0].mean(axis=0)
    u_mean = gates[..., 1].mean(axis=0)
    w_std = gates[..., 0].std(axis=0)
    u_std = gates[..., 1].std(axis=0)

    x = np.arange(gates.shape[1])
    step_labels = timesteps[0]

    plt.figure(figsize=(10, 4))
    plt.plot(x, w_mean, label="w left conditional weight")
    plt.fill_between(x, w_mean - w_std, w_mean + w_std, alpha=0.2)
    plt.plot(x, u_mean, label="u right conditional weight")
    plt.fill_between(x, u_mean - u_std, u_mean + u_std, alpha=0.2)
    plt.xlabel("diffusion iteration")
    plt.ylabel("gate value")
    plt.title("Factorized gates over denoising")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig(args.output_dir / "01_gates_over_diffusion.png")

    tick_idx = np.linspace(0, len(step_labels) - 1, num=8, dtype=int)
    plt.figure(figsize=(10, 4))
    plt.plot(step_labels, w_mean, label="w left")
    plt.plot(step_labels, u_mean, label="u right")
    plt.gca().invert_xaxis()
    plt.xticks(step_labels[tick_idx])
    plt.xlabel("scheduler timestep")
    plt.ylabel("gate value")
    plt.title("Gates vs scheduler timestep")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig(args.output_dir / "02_gates_vs_scheduler_timestep.png")

    left_cond_mse = mse(left_cond, left_marginal, axis=(2, 3)).mean(axis=0)
    right_cond_mse = mse(right_cond, right_marginal, axis=(2, 3)).mean(axis=0)
    left_pred_marginal_mse = mse(left_pred, left_marginal, axis=(2, 3)).mean(axis=0)
    right_pred_marginal_mse = mse(right_pred, right_marginal, axis=(2, 3)).mean(axis=0)
    left_pred_cond_mse = mse(left_pred, left_cond, axis=(2, 3)).mean(axis=0)
    right_pred_cond_mse = mse(right_pred, right_cond, axis=(2, 3)).mean(axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(x, left_cond_mse, label="left cond vs marginal")
    plt.plot(x, right_cond_mse, label="right cond vs marginal")
    plt.plot(x, left_pred_marginal_mse, "--", label="left final vs marginal")
    plt.plot(x, right_pred_marginal_mse, "--", label="right final vs marginal")
    plt.xlabel("diffusion iteration")
    plt.ylabel("MSE")
    plt.title("Conditional branch difference")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig(args.output_dir / "03_branch_mse_over_diffusion.png")

    plt.figure(figsize=(8, 4))
    names = [
        "L cond-marg",
        "R cond-marg",
        "L final-marg",
        "R final-marg",
        "L final-cond",
        "R final-cond",
    ]
    vals = [
        mse(left_cond, left_marginal),
        mse(right_cond, right_marginal),
        mse(left_pred, left_marginal),
        mse(right_pred, right_marginal),
        mse(left_pred, left_cond),
        mse(right_pred, right_cond),
    ]
    plt.bar(names, vals)
    plt.ylabel("global MSE")
    plt.title("Global branch differences")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    savefig(args.output_dir / "04_global_branch_mse.png")

    call_idx = min(max(args.policy_call, 0), left_pred.shape[0] - 1)
    diff_idx = args.diffusion_index
    if diff_idx < 0:
        diff_idx = left_pred.shape[1] + diff_idx
    diff_idx = min(max(diff_idx, 0), left_pred.shape[1] - 1)

    def plot_arm_trajectories(prefix, marginal, cond, pred):
        horizon = np.arange(marginal.shape[2])
        fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
        axes = axes.reshape(-1)
        for dim in range(marginal.shape[3]):
            ax = axes[dim]
            ax.plot(horizon, marginal[call_idx, diff_idx, :, dim], label="marginal")
            ax.plot(horizon, cond[call_idx, diff_idx, :, dim], label="conditional")
            ax.plot(horizon, pred[call_idx, diff_idx, :, dim], label="final", linestyle="--")
            ax.set_title(f"{prefix} dim {dim}")
            ax.grid(alpha=0.3)
        axes[-1].axis("off")
        axes[0].legend()
        fig.suptitle(f"{prefix} trajectories call={call_idx}, diffusion_index={diff_idx}, timestep={timesteps[call_idx, diff_idx]}")
        savefig(args.output_dir / f"05_{prefix}_trajectories.png")

    plot_arm_trajectories("left", left_marginal, left_cond, left_pred)
    plot_arm_trajectories("right", right_marginal, right_cond, right_pred)

    action_summary = []
    if "marginal_left_action" in data.files:
        marginal_left_action = data["marginal_left_action"][:, 0, :, :]
        marginal_right_action = data["marginal_right_action"][:, 0, :, :]
        conditional_left_action = data["conditional_left_action"][:, 0, :, :]
        conditional_right_action = data["conditional_right_action"][:, 0, :, :]
        final_left_action = data["final_left_action"][:, 0, :, :]
        final_right_action = data["final_right_action"][:, 0, :, :]

        action_names = [
            "L cond-marg",
            "R cond-marg",
            "L final-marg",
            "R final-marg",
            "L final-cond",
            "R final-cond",
        ]
        action_vals = [
            mse(conditional_left_action, marginal_left_action),
            mse(conditional_right_action, marginal_right_action),
            mse(final_left_action, marginal_left_action),
            mse(final_right_action, marginal_right_action),
            mse(final_left_action, conditional_left_action),
            mse(final_right_action, conditional_right_action),
        ]
        plt.figure(figsize=(8, 4))
        plt.bar(action_names, action_vals)
        plt.ylabel("global MSE in action space")
        plt.title("Denoised action branch differences")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.3)
        savefig(args.output_dir / "06_denoised_action_global_mse.png")

        def plot_action_trajectories(prefix, marginal, conditional, final):
            horizon = np.arange(marginal.shape[1])
            fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
            axes = axes.reshape(-1)
            for dim in range(marginal.shape[2]):
                ax = axes[dim]
                ax.plot(horizon, marginal[call_idx, :, dim], label="marginal action")
                ax.plot(horizon, conditional[call_idx, :, dim], label="conditional action")
                ax.plot(horizon, final[call_idx, :, dim], label="final action", linestyle="--")
                ax.set_title(f"{prefix} action dim {dim}")
                ax.grid(alpha=0.3)
            axes[-1].axis("off")
            axes[0].legend()
            fig.suptitle(f"Denoised {prefix} action trajectories call={call_idx}")
            savefig(args.output_dir / f"07_{prefix}_denoised_action_trajectories.png")

        plot_action_trajectories("left", marginal_left_action, conditional_left_action, final_left_action)
        plot_action_trajectories("right", marginal_right_action, conditional_right_action, final_right_action)

        action_start = args.action_start
        action_end = action_start + args.action_steps

        def concat_episode_actions(action):
            return action[:, action_start:action_end, :].reshape(-1, action.shape[-1])

        episode_left_marginal = concat_episode_actions(marginal_left_action)
        episode_left_conditional = concat_episode_actions(conditional_left_action)
        episode_left_final = concat_episode_actions(final_left_action)
        episode_right_marginal = concat_episode_actions(marginal_right_action)
        episode_right_conditional = concat_episode_actions(conditional_right_action)
        episode_right_final = concat_episode_actions(final_right_action)
        episode_steps = np.arange(episode_left_final.shape[0])

        left_episode_cond_marg = mse(episode_left_conditional, episode_left_marginal, axis=1)
        right_episode_cond_marg = mse(episode_right_conditional, episode_right_marginal, axis=1)
        left_episode_final_marg = mse(episode_left_final, episode_left_marginal, axis=1)
        right_episode_final_marg = mse(episode_right_final, episode_right_marginal, axis=1)
        left_episode_final_cond = mse(episode_left_final, episode_left_conditional, axis=1)
        right_episode_final_cond = mse(episode_right_final, episode_right_conditional, axis=1)

        plt.figure(figsize=(12, 5))
        plt.plot(episode_steps, left_episode_cond_marg, label="left cond vs marginal")
        plt.plot(episode_steps, right_episode_cond_marg, label="right cond vs marginal")
        plt.plot(episode_steps, left_episode_final_marg, "--", label="left final vs marginal")
        plt.plot(episode_steps, right_episode_final_marg, "--", label="right final vs marginal")
        plt.xlabel("executed action step in saved trajectory")
        plt.ylabel("MSE over arm dims")
        plt.title("Denoised action differences over episode trajectory")
        plt.legend()
        plt.grid(alpha=0.3)
        savefig(args.output_dir / "08_episode_action_mse_over_steps.png")

        n_steps = len(episode_steps)
        thirds = [
            (0, n_steps // 3, "early"),
            (n_steps // 3, 2 * n_steps // 3, "middle"),
            (2 * n_steps // 3, n_steps, "late"),
        ]
        phase_names = [name for _, _, name in thirds]
        left_phase_mean = [left_episode_cond_marg[s:e].mean() for s, e, _ in thirds]
        right_phase_mean = [right_episode_cond_marg[s:e].mean() for s, e, _ in thirds]
        left_phase_max = [left_episode_cond_marg[s:e].max() for s, e, _ in thirds]
        right_phase_max = [right_episode_cond_marg[s:e].max() for s, e, _ in thirds]

        x_phase = np.arange(len(phase_names))
        width = 0.35
        plt.figure(figsize=(8, 4))
        plt.bar(x_phase - width / 2, left_phase_mean, width, label="left mean")
        plt.bar(x_phase + width / 2, right_phase_mean, width, label="right mean")
        plt.xticks(x_phase, phase_names)
        plt.ylabel("MSE over arm dims")
        plt.title("Cond vs marginal difference by episode phase")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        savefig(args.output_dir / "11_episode_phase_cond_marg_mean.png")

        plt.figure(figsize=(8, 4))
        plt.bar(x_phase - width / 2, left_phase_max, width, label="left max")
        plt.bar(x_phase + width / 2, right_phase_max, width, label="right max")
        plt.xticks(x_phase, phase_names)
        plt.ylabel("MSE over arm dims")
        plt.title("Cond vs marginal peak by episode phase")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        savefig(args.output_dir / "12_episode_phase_cond_marg_max.png")

        combined_cond_marg = 0.5 * (left_episode_cond_marg + right_episode_cond_marg)
        top_idx = np.argsort(combined_cond_marg)[-10:][::-1]
        plt.figure(figsize=(12, 5))
        plt.plot(episode_steps, combined_cond_marg, label="mean(left,right) cond vs marginal")
        plt.scatter(top_idx, combined_cond_marg[top_idx], color="red", label="top peaks")
        for idx in top_idx[:5]:
            plt.annotate(f"{idx}", (idx, combined_cond_marg[idx]), textcoords="offset points", xytext=(0, 6), ha="center")
        plt.xlabel("executed action step in saved trajectory")
        plt.ylabel("MSE over arm dims")
        plt.title("Top cond-vs-marginal difference peaks over episode")
        plt.legend()
        plt.grid(alpha=0.3)
        savefig(args.output_dir / "13_episode_cond_marg_peaks.png")

        plt.figure(figsize=(12, 5))
        plt.plot(episode_steps, left_episode_final_marg, label="left final vs marginal")
        plt.plot(episode_steps, left_episode_final_cond, label="left final vs conditional")
        plt.plot(episode_steps, right_episode_final_marg, label="right final vs marginal")
        plt.plot(episode_steps, right_episode_final_cond, label="right final vs conditional")
        plt.xlabel("executed action step in saved trajectory")
        plt.ylabel("MSE over arm dims")
        plt.title("Whether final action is closer to marginal or conditional")
        plt.legend()
        plt.grid(alpha=0.3)
        savefig(args.output_dir / "09_episode_final_closeness_over_steps.png")

        def plot_episode_arm(prefix, marginal, conditional, final):
            fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
            axes = axes.reshape(-1)
            for dim in range(marginal.shape[1]):
                ax = axes[dim]
                ax.plot(episode_steps, marginal[:, dim], label="marginal")
                ax.plot(episode_steps, conditional[:, dim], label="conditional")
                ax.plot(episode_steps, final[:, dim], label="final", linestyle="--")
                ax.set_title(f"{prefix} action dim {dim}")
                ax.grid(alpha=0.3)
            axes[-1].axis("off")
            axes[0].legend()
            fig.suptitle(f"Episode denoised {prefix} actions: marginal vs conditional vs final")
            savefig(args.output_dir / f"10_episode_{prefix}_actions.png")

        plot_episode_arm("left", episode_left_marginal, episode_left_conditional, episode_left_final)
        plot_episode_arm("right", episode_right_marginal, episode_right_conditional, episode_right_final)

        left_final_closer_marginal = left_episode_final_marg < left_episode_final_cond
        right_final_closer_marginal = right_episode_final_marg < right_episode_final_cond

        action_summary.extend([
            "",
            "Denoised action-space statistics:",
            f"left action cond vs marginal mse: {action_vals[0]:.8f}",
            f"right action cond vs marginal mse: {action_vals[1]:.8f}",
            f"left action final vs marginal mse: {action_vals[2]:.8f}",
            f"right action final vs marginal mse: {action_vals[3]:.8f}",
            f"left action final vs conditional mse: {action_vals[4]:.8f}",
            f"right action final vs conditional mse: {action_vals[5]:.8f}",
            "",
            "Episode executed action-space statistics:",
            f"executed action steps saved: {episode_left_final.shape[0]}",
            f"action slice per call: [{action_start}:{action_end}]",
            f"left episode cond vs marginal mse mean/max: {left_episode_cond_marg.mean():.8f} / {left_episode_cond_marg.max():.8f}",
            f"right episode cond vs marginal mse mean/max: {right_episode_cond_marg.mean():.8f} / {right_episode_cond_marg.max():.8f}",
            f"left episode final vs marginal mse mean/max: {left_episode_final_marg.mean():.8f} / {left_episode_final_marg.max():.8f}",
            f"right episode final vs marginal mse mean/max: {right_episode_final_marg.mean():.8f} / {right_episode_final_marg.max():.8f}",
            f"left final closer to marginal steps: {left_final_closer_marginal.sum()} / {len(left_final_closer_marginal)}",
            f"right final closer to marginal steps: {right_final_closer_marginal.sum()} / {len(right_final_closer_marginal)}",
            f"left cond-marg phase mean early/middle/late: {left_phase_mean[0]:.8f} / {left_phase_mean[1]:.8f} / {left_phase_mean[2]:.8f}",
            f"right cond-marg phase mean early/middle/late: {right_phase_mean[0]:.8f} / {right_phase_mean[1]:.8f} / {right_phase_mean[2]:.8f}",
            f"left cond-marg phase max early/middle/late: {left_phase_max[0]:.8f} / {left_phase_max[1]:.8f} / {left_phase_max[2]:.8f}",
            f"right cond-marg phase max early/middle/late: {right_phase_max[0]:.8f} / {right_phase_max[1]:.8f} / {right_phase_max[2]:.8f}",
            "top combined cond-marg peaks: "
            + ", ".join([
                f"step {idx} (call {idx // args.action_steps}, in_call {idx % args.action_steps}, mse {combined_cond_marg[idx]:.8f})"
                for idx in top_idx[:10]
            ]),
        ])

    summary = []
    summary.append(f"debug_file: {args.debug_file}")
    summary.append(f"policy_calls: {left_pred.shape[0]}")
    summary.append(f"diffusion_steps: {left_pred.shape[1]}")
    summary.append(f"horizon: {left_pred.shape[2]}")
    summary.append(f"arm_dim: {left_pred.shape[3]}")
    summary.append("")
    summary.append(f"w mean/min/max: {gates[..., 0].mean():.6f} / {gates[..., 0].min():.6f} / {gates[..., 0].max():.6f}")
    summary.append(f"u mean/min/max: {gates[..., 1].mean():.6f} / {gates[..., 1].min():.6f} / {gates[..., 1].max():.6f}")
    summary.append("")
    summary.append(f"left cond vs marginal mse: {mse(left_cond, left_marginal):.8f}")
    summary.append(f"right cond vs marginal mse: {mse(right_cond, right_marginal):.8f}")
    summary.append(f"left final vs marginal mse: {mse(left_pred, left_marginal):.8f}")
    summary.append(f"right final vs marginal mse: {mse(right_pred, right_marginal):.8f}")
    summary.append(f"left final vs conditional mse: {mse(left_pred, left_cond):.8f}")
    summary.append(f"right final vs conditional mse: {mse(right_pred, right_cond):.8f}")
    summary.extend(action_summary)
    summary.append("")
    summary.append(f"plots saved to: {args.output_dir}")
    (args.output_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
