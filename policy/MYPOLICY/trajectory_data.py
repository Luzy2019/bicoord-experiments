import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


LEFT_DIM = 7
RIGHT_DIM = 7
ACTION_DIM = LEFT_DIM + RIGHT_DIM


def _smooth_segment(rng: np.random.Generator, length: int, dim: int) -> np.ndarray:
    """Generate a smooth action segment with zero-ish endpoints."""
    time = np.linspace(0.0, 1.0, length, dtype=np.float32)
    segment = np.zeros((length, dim), dtype=np.float32)
    for action_idx in range(dim):
        amp1 = rng.uniform(0.15, 1.0)
        amp2 = rng.uniform(0.0, 0.35)
        phase1 = rng.uniform(0.0, 2.0 * np.pi)
        phase2 = rng.uniform(0.0, 2.0 * np.pi)
        freq1 = rng.integers(1, 4)
        freq2 = rng.integers(1, 3)
        curve = (
            amp1 * np.sin(freq1 * np.pi * time + phase1)
            + amp2 * np.sin(freq2 * 2.0 * np.pi * time + phase2)
        )
        # Float roundoff can make sin(pi) slightly negative at the endpoint;
        # clamp before the fractional power to avoid writing NaNs into data.
        envelope = np.clip(np.sin(np.pi * time), 0.0, None) ** 1.5
        segment[:, action_idx] = curve * envelope
    return segment


def _correlate_right_segment(
    rng: np.random.Generator,
    left_segment: np.ndarray,
    right_length: int,
    correlated: bool,
) -> np.ndarray:
    if not correlated:
        return _smooth_segment(rng, right_length, RIGHT_DIM)

    source_time = np.linspace(0.0, 1.0, left_segment.shape[0], dtype=np.float32)
    target_time = np.linspace(0.0, 1.0, right_length, dtype=np.float32)
    resampled_left = np.stack(
        [np.interp(target_time, source_time, left_segment[:, dim]) for dim in range(LEFT_DIM)],
        axis=-1,
    ).astype(np.float32)
    mixing = rng.normal(0.0, 0.35, size=(LEFT_DIM, RIGHT_DIM)).astype(np.float32)
    right = resampled_left @ mixing
    right += 0.25 * _smooth_segment(rng, right_length, RIGHT_DIM)
    return right.astype(np.float32)


def _place_segment(trajectory: np.ndarray, start: int, values: np.ndarray, slc: slice) -> None:
    end = min(start + values.shape[0], trajectory.shape[0])
    if end > start:
        trajectory[start:end, slc] = values[: end - start]


def _active_window(arm_actions: np.ndarray, eps: float = 1e-6) -> Tuple[int, int]:
    active = np.linalg.norm(arm_actions, axis=-1) > eps
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return 0, 0
    return int(indices[0]), int(indices[-1]) + 1


def parallelize_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Shift each arm's active segment to the start of the trajectory.

    The source demonstration can be sequential, e.g. left moves in [0, 50) and
    right moves in [50, 100). The returned target keeps each arm's local action
    curve but aligns both active windows at t=0, producing a parallel rollout.
    """
    result = np.zeros_like(trajectory)
    for slc in (slice(0, LEFT_DIM), slice(LEFT_DIM, ACTION_DIM)):
        start, end = _active_window(trajectory[:, slc])
        if end > start:
            result[: end - start, slc] = trajectory[start:end, slc]
    return result


def generate_dataset(
    num_samples: int = 4096,
    horizon: int = 100,
    segment_min: int = 35,
    segment_max: int = 55,
    gap_min: int = 0,
    gap_max: int = 12,
    correlated_prob: float = 0.5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sources = np.zeros((num_samples, horizon, ACTION_DIM), dtype=np.float32)
    targets = np.zeros_like(sources)
    meta = np.zeros((num_samples, 4), dtype=np.int32)

    for sample_idx in range(num_samples):
        left_len = int(rng.integers(segment_min, segment_max + 1))
        right_len = int(rng.integers(segment_min, segment_max + 1))
        gap = int(rng.integers(gap_min, gap_max + 1))
        correlated = bool(rng.random() < correlated_prob)

        left_segment = _smooth_segment(rng, left_len, LEFT_DIM)
        right_segment = _correlate_right_segment(rng, left_segment, right_len, correlated)

        left_start = 0
        right_start = min(left_len + gap, horizon - 1)
        source = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
        _place_segment(source, left_start, left_segment, slice(0, LEFT_DIM))
        _place_segment(source, right_start, right_segment, slice(LEFT_DIM, ACTION_DIM))

        sources[sample_idx] = source
        targets[sample_idx] = parallelize_trajectory(source)
        meta[sample_idx] = [left_len, right_len, gap, int(correlated)]

    return {
        "source": sources,
        "target": targets,
        "meta": meta,
    }


def save_dataset(path: Path, data: Dict[str, np.ndarray], config: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    path.with_suffix(".json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic bimanual trajectory data.")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_bimanual_fm.npz"))
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--segment-min", type=int, default=35)
    parser.add_argument("--segment-max", type=int, default=55)
    parser.add_argument("--gap-min", type=int, default=0)
    parser.add_argument("--gap-max", type=int, default=12)
    parser.add_argument("--correlated-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = vars(args).copy()
    config["output"] = str(args.output)
    data = generate_dataset(
        num_samples=args.num_samples,
        horizon=args.horizon,
        segment_min=args.segment_min,
        segment_max=args.segment_max,
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        correlated_prob=args.correlated_prob,
        seed=args.seed,
    )
    save_dataset(args.output, data, config)
    print(f"Saved {data['source'].shape[0]} samples to {args.output}")
    print(f"source shape: {data['source'].shape}, target shape: {data['target'].shape}")


if __name__ == "__main__":
    main()
