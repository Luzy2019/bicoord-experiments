import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import zarr


def parse_args():
    parser = argparse.ArgumentParser(description="Export camera frames from a RoboTwin zarr dataset to mp4.")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("data/handover_block_with_bowls-demo_clean-50.zarr"),
        help="Path to the zarr dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_inspect_mp4"),
        help="Directory to save exported videos.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="head_camera",
        help="Camera key under zarr/data, e.g. head_camera.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode id to export. If omitted, exports all episodes.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Output video fps.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap when exporting all episodes.",
    )
    return parser.parse_args()


def to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected 3D image frame, got shape {frame.shape}")

    # Training data may be saved as CHW, while imageio expects HWC.
    if frame.shape[0] in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)

    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def episode_range(episode_ends: np.ndarray, episode_id: int):
    if episode_id < 0 or episode_id >= len(episode_ends):
        raise IndexError(f"episode must be in [0, {len(episode_ends) - 1}], got {episode_id}")
    start = 0 if episode_id == 0 else int(episode_ends[episode_id - 1])
    end = int(episode_ends[episode_id])
    return start, end


def export_episode(camera_data, episode_ends, episode_id: int, output_dir: Path, fps: int):
    start, end = episode_range(episode_ends, episode_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"episode{episode_id}.mp4"

    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for idx in range(start, end):
            writer.append_data(to_hwc_uint8(camera_data[idx]))

    print(f"saved {output_path} frames={end - start}")


def main():
    args = parse_args()
    root = zarr.open(str(args.zarr_path), mode="r")
    data = root["data"]
    if args.camera_key not in data:
        raise KeyError(f"{args.camera_key} not found. Available data keys: {list(data.keys())}")

    episode_ends = root["meta"]["episode_ends"][:]
    camera_data = data[args.camera_key]

    print(f"zarr: {args.zarr_path}")
    print(f"camera: {args.camera_key}, shape={camera_data.shape}, dtype={camera_data.dtype}")
    print(f"episodes: {len(episode_ends)}")

    if args.episode is not None:
        export_episode(camera_data, episode_ends, args.episode, args.output_dir, args.fps)
        return

    episode_ids = range(len(episode_ends))
    if args.max_episodes is not None:
        episode_ids = range(min(args.max_episodes, len(episode_ends)))

    for episode_id in episode_ids:
        export_episode(camera_data, episode_ends, episode_id, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
