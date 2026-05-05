import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect saved factorized bimanual DP debug npz files.")
    parser.add_argument("debug_file", type=Path, help="Path to factorized_step_*.npz")
    return parser.parse_args()


def mse(a, b):
    return np.mean((a - b) ** 2)


def main():
    args = parse_args()
    data = np.load(args.debug_file)

    timesteps = data["timesteps"]
    gates = data["factorized_gates"]
    left_marginal = data["left_marginal"]
    right_marginal = data["right_marginal"]
    left_cond = data["left_cond"]
    right_cond = data["right_cond"]
    left_pred = data["left_pred"]
    right_pred = data["right_pred"]

    print(f"file: {args.debug_file}")
    print(f"policy calls: {data.get('num_policy_calls', np.asarray(timesteps.shape[0]))}")
    print(f"timesteps: {timesteps.shape}, first={timesteps.reshape(-1)[0]}, last={timesteps.reshape(-1)[-1]}")
    print(f"gates shape: {gates.shape}  # [policy_call, diffusion_step, batch, 2]")
    print(f"w(left cond weight) mean/min/max: {gates[..., 0].mean():.4f} / {gates[..., 0].min():.4f} / {gates[..., 0].max():.4f}")
    print(f"u(right cond weight) mean/min/max: {gates[..., 1].mean():.4f} / {gates[..., 1].min():.4f} / {gates[..., 1].max():.4f}")
    print()
    print("branch differences:")
    print(f"left_cond vs left_marginal mse: {mse(left_cond, left_marginal):.6f}")
    print(f"right_cond vs right_marginal mse: {mse(right_cond, right_marginal):.6f}")
    print(f"left_pred vs left_marginal mse: {mse(left_pred, left_marginal):.6f}")
    print(f"right_pred vs right_marginal mse: {mse(right_pred, right_marginal):.6f}")
    print(f"left_pred vs left_cond mse: {mse(left_pred, left_cond):.6f}")
    print(f"right_pred vs right_cond mse: {mse(right_pred, right_cond):.6f}")


if __name__ == "__main__":
    main()
