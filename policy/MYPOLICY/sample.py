"""Sample MYPOLICY trajectories from a stage-1 or stage-2 checkpoint.

Stage-1 checkpoints only contain the base flow:
    bash sample.sh stage1 -> writes  outputs/mypolicy_samples_stage1.npz
                              keys:   source, target, prediction (== raw FM)

Stage-2 checkpoints contain base flow + per-arm warp head:
    bash sample.sh stage2 -> writes  outputs/mypolicy_samples_stage2.npz
                              keys:   source, target, prediction_raw, prediction (warped)

"target" is only loaded from the dataset for evaluation/visualization. It is
NEVER used during training.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from policy.MYPOLICY.model import (
        ACTION_DIM,
        AsymmetricArmWarpHead,
        Normalizer,
        SpeedModulatedPolicy,
        TrajectoryFlowMatchingPolicy,
    )
    from policy.MYPOLICY.trajectory_data import generate_dataset
else:
    from .model import (
        ACTION_DIM,
        AsymmetricArmWarpHead,
        Normalizer,
        SpeedModulatedPolicy,
        TrajectoryFlowMatchingPolicy,
    )
    from .trajectory_data import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample parallel bimanual trajectories from a trained model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mypolicy_base_best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_bimanual_fm.npz"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/mypolicy_samples.npz"))
    parser.add_argument(
        "--mode",
        choices=["generate", "warp"],
        default="generate",
        help="generate: sample from base flow then optionally warp. warp: warp real source samples (stage 2 only).",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    stage = int(payload.get("stage", 1))
    action_dim = cfg.get("action_dim", ACTION_DIM)
    base_flow = TrajectoryFlowMatchingPolicy(
        action_dim=action_dim,
        hidden_dim=cfg.get("hidden_dim", 256),
        num_blocks=cfg.get("num_blocks", 6),
    ).to(device)

    if stage == 1:
        base_flow.load_state_dict(payload["model_state"])
    else:
        base_flow.load_state_dict(payload["base_flow_state"])

    warp_head = None
    policy = None
    if stage == 2:
        warp_head = AsymmetricArmWarpHead(
            action_dim=action_dim,
            hidden_dim=cfg.get("warp_hidden_dim", 128),
            alpha_max=cfg.get("warp_alpha_max", 3.0),
            max_shift_ratio=cfg.get("warp_max_shift_ratio", 0.6),
        ).to(device)
        warp_head.load_state_dict(payload["warp_head_state"])
        policy = SpeedModulatedPolicy(base_flow=base_flow, warp_head=warp_head).to(device)
        policy.eval()

    base_flow.eval()
    norms = payload["normalizers"]
    action_mean = norms.get("action_mean", norms.get("condition_mean"))
    action_std = norms.get("action_std", norms.get("condition_std"))
    normalizers = {
        "action": Normalizer(action_mean.to(device), action_std.to(device)),
    }
    return {
        "stage": stage,
        "base_flow": base_flow,
        "warp_head": warp_head,
        "policy": policy,
        "normalizers": normalizers,
        "config": cfg,
    }


def load_source_target(path: Path, index: int, num_samples: int):
    if path.exists():
        payload = np.load(path)
        source = payload["source"]
        target = payload["target"] if "target" in payload.files else np.zeros_like(source)
    else:
        generated = generate_dataset(num_samples=max(index + num_samples, num_samples), seed=123)
        source = generated["source"]
        target = generated["target"]
    end = min(index + num_samples, source.shape[0])
    return source[index:end], target[index:end]


def active_overlap(trajectory: np.ndarray, eps: float = 1e-4) -> float:
    left = np.linalg.norm(trajectory[..., :7], axis=-1) > eps
    right = np.linalg.norm(trajectory[..., 7:], axis=-1) > eps
    union = np.logical_or(left, right).sum(axis=-1).clip(min=1)
    overlap = np.logical_and(left, right).sum(axis=-1)
    return float(np.mean(overlap / union))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    info = load_checkpoint(args.checkpoint, device)
    stage = info["stage"]
    base_flow = info["base_flow"]
    policy = info["policy"]
    normalizers = info["normalizers"]
    cfg = info["config"]

    source_np, target_np = load_source_target(args.data, args.index, args.num_samples)
    batch_size = source_np.shape[0]
    horizon = source_np.shape[1]
    steps = args.steps or cfg.get("sample_steps", 100)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    save_payload = {"source": source_np, "target": target_np}

    unnormalize_fn = normalizers["action"].unnormalize

    if args.mode == "warp":
        if stage != 2:
            raise ValueError("Mode 'warp' requires a stage-2 checkpoint.")
        # The warp head was trained on raw (un-normalized) source so it can
        # detect zero-padded idle regions. Feed it raw source directly.
        source_tensor = torch.from_numpy(source_np).float().to(device)
        outputs = policy.warp_only(source_tensor)
        save_payload["prediction_raw"] = outputs["raw"].cpu().numpy()
        save_payload["prediction"] = outputs["warped"].cpu().numpy()
    elif stage == 1:
        raw = base_flow.sample(
            batch_size=batch_size,
            horizon=horizon,
            device=device,
            dtype=torch.float32,
            num_steps=steps,
            generator=generator,
        )
        raw_np = unnormalize_fn(raw).cpu().numpy()
        save_payload["prediction"] = raw_np
    else:
        outputs = policy.sample(
            batch_size=batch_size,
            horizon=horizon,
            device=device,
            dtype=torch.float32,
            num_steps=steps,
            generator=generator,
            unnormalize=unnormalize_fn,
        )
        save_payload["prediction_raw"] = outputs["raw"].cpu().numpy()
        save_payload["prediction"] = outputs["warped"].cpu().numpy()
        print(
            f"shift_L={float(outputs['shift_L'].mean()):.2f} "
            f"scale_L={float(outputs['scale_L'].mean()):.2f} "
            f"shift_R={float(outputs['shift_R'].mean()):.2f} "
            f"scale_R={float(outputs['scale_R'].mean()):.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **save_payload)
    print(f"Saved samples to {args.output}")
    print(f"source        active overlap: {active_overlap(source_np):.3f}")
    print(f"target (ref)  active overlap: {active_overlap(target_np):.3f}")
    if "prediction_raw" in save_payload:
        print(f"prediction_raw  active overlap: {active_overlap(save_payload['prediction_raw']):.3f}")
    print(f"prediction    active overlap: {active_overlap(save_payload['prediction']):.3f}")


if __name__ == "__main__":
    main()
