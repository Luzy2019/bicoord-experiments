import argparse
from pathlib import Path

import numpy as np


def make_trajectories(num_trajectories: int, trajectory_len: int, action_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 1.0, trajectory_len, dtype=np.float32)
    trajectories = []

    for _ in range(num_trajectories):
        traj = np.zeros((trajectory_len, action_dim), dtype=np.float32)
        base_phase = rng.uniform(0.0, 2.0 * np.pi, size=(action_dim,))
        base_freq = rng.uniform(0.5, 2.5, size=(action_dim,))
        amplitude = rng.uniform(0.2, 1.0, size=(action_dim,))
        trend = rng.uniform(-0.5, 0.5, size=(action_dim,))

        for dim in range(action_dim):
            wave = amplitude[dim] * np.sin(2.0 * np.pi * base_freq[dim] * time + base_phase[dim])
            smooth_trend = trend[dim] * (time - 0.5)
            traj[:, dim] = wave + smooth_trend

        traj += rng.normal(0.0, 0.02, size=traj.shape).astype(np.float32)
        trajectories.append(traj)

    return np.stack(trajectories, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Generate toy action trajectories for DP/FM demos.")
    parser.add_argument("--num-trajectories", type=int, default=256)
    parser.add_argument("--trajectory-len", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("test_trajectories.npz"))
    args = parser.parse_args()

    actions = make_trajectories(args.num_trajectories, args.trajectory_len, args.action_dim, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        actions=actions,
        trajectory_len=np.array(args.trajectory_len, dtype=np.int64),
        chunk_size=np.array(args.chunk_size, dtype=np.int64),
        action_dim=np.array(args.action_dim, dtype=np.int64),
    )
    print(f"saved {args.output}")
    print(f"actions shape: {actions.shape}, chunk_size: {args.chunk_size}")


if __name__ == "__main__":
    main()
