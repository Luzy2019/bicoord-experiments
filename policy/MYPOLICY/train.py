"""MYPOLICY staged training.

Stage 1 (base flow):
    bash train.sh 1                # -> checkpoints/mypolicy_base.pt
Stage 2 (asymmetric per-arm warp head, frozen base flow):
    bash train.sh 2                # -> checkpoints/mypolicy_speed.pt

Both stages train ONLY on source trajectories. No target labels are loaded or
constructed offline. Stage 2 self-supervised losses are computed online from
the source batch alone (see `asymmetric_warp_loss` for details).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from policy.MYPOLICY.model import (
        ACTION_DIM,
        AsymmetricArmWarpHead,
        SpeedModulatedPolicy,
        TrajectoryFlowMatchingPolicy,
        build_normalizers,
    )
    from policy.MYPOLICY.trajectory_data import generate_dataset, save_dataset
else:
    from .model import (
        ACTION_DIM,
        AsymmetricArmWarpHead,
        SpeedModulatedPolicy,
        TrajectoryFlowMatchingPolicy,
        build_normalizers,
    )
    from .trajectory_data import generate_dataset, save_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MYPOLICY (staged FM + per-arm warp).")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1, help="1: base FM. 2: per-arm warp head.")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_bimanual_fm.npz"))
    parser.add_argument("--generate-if-missing", action="store_true", default=True)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output checkpoint (defaults: stage 1 -> checkpoints/mypolicy_base.pt, stage 2 -> mypolicy_speed.pt).",
    )
    parser.add_argument(
        "--base-ckpt",
        type=Path,
        default=Path("checkpoints/mypolicy_base.pt"),
        help="Required for --stage 2. Path to the stage-1 base flow checkpoint.",
    )
    parser.add_argument("--warp-hidden-dim", type=int, default=128)
    parser.add_argument("--warp-alpha-max", type=float, default=3.0)
    parser.add_argument("--warp-max-shift-ratio", type=float, default=0.6)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--compactness-weight", type=float, default=1.0)
    parser.add_argument("--content-weight", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=5.0)
    parser.add_argument("--fast-weight", type=float, default=0.0)
    parser.add_argument("--scale-reg-weight", type=float, default=1.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_generate_dataset(args: argparse.Namespace) -> torch.Tensor:
    if not args.data.exists():
        if not args.generate_if_missing:
            raise FileNotFoundError(args.data)
        config = {
            "num_samples": args.num_samples,
            "horizon": args.horizon,
            "seed": args.seed,
        }
        data = generate_dataset(
            num_samples=args.num_samples,
            horizon=args.horizon,
            seed=args.seed,
        )
        save_dataset(args.data, data, config)
        print(f"Generated dataset: {args.data}")

    payload = np.load(args.data)
    source = torch.from_numpy(payload["source"]).float()
    if not torch.isfinite(source).all():
        raise ValueError(
            f"Dataset contains NaN/Inf: {args.data}. "
            "Regenerate it with `bash generate_data.sh` after the trajectory_data.py fix."
        )
    return source


def default_output(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    if args.stage == 1:
        return Path("checkpoints/mypolicy_base.pt")
    return Path("checkpoints/mypolicy_speed.pt")


def save_checkpoint(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_stage1(args: argparse.Namespace) -> None:
    source = load_or_generate_dataset(args)
    normalizers = build_normalizers(source)
    normalized_check = normalizers["action"].normalize(source)
    if not torch.isfinite(normalized_check).all():
        raise ValueError("Normalized source dataset contains NaN/Inf; check source statistics.")

    loader = DataLoader(
        TensorDataset(source),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    device = torch.device(args.device)
    model = TrajectoryFlowMatchingPolicy(
        action_dim=source.shape[-1],
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    output = default_output(args)
    log_path = output.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for (source_raw,) in loader:
                source_raw = source_raw.to(device)
                source_norm = normalizers["action"].normalize(source_raw)
                loss = model.compute_loss(source_norm)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite loss detected during stage 1. Regenerate the dataset and retry."
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            epoch_loss = float(np.mean(losses))
            row = {"stage": 1, "epoch": epoch, "flow_loss": epoch_loss, "lr": args.lr}
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            print(f"[stage1] epoch {epoch:04d} | flow_loss={epoch_loss:.6f}")

            payload_common = {
                "stage": 1,
                "model_state": model.state_dict(),
                "epoch": epoch,
                "flow_loss": epoch_loss,
                "config": {
                    "action_dim": source.shape[-1],
                    "hidden_dim": args.hidden_dim,
                    "num_blocks": args.num_blocks,
                    "sample_steps": args.sample_steps,
                },
                "normalizers": {
                    "action_mean": normalizers["action"].mean.cpu(),
                    "action_std": normalizers["action"].std.cpu(),
                },
                "args": vars(args),
                "training": {
                    "uses_offline_target": False,
                    "data": "source-only rectified flow on the source distribution",
                },
            }

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(output.with_name(output.stem + "_best.pt"), payload_common)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(output, payload_common)

    print(f"[stage1] Saved checkpoint: {output}")
    print(f"[stage1] Saved log: {log_path}")


def load_stage1_into(args: argparse.Namespace, device: torch.device, action_dim: int):
    if not args.base_ckpt.exists():
        raise FileNotFoundError(
            f"Stage 2 requires a base-flow checkpoint at {args.base_ckpt}. "
            "Train stage 1 first with `bash train.sh 1`."
        )
    payload = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    base_flow = TrajectoryFlowMatchingPolicy(
        action_dim=cfg.get("action_dim", action_dim),
        hidden_dim=cfg.get("hidden_dim", args.hidden_dim),
        num_blocks=cfg.get("num_blocks", args.num_blocks),
    ).to(device)
    base_flow.load_state_dict(payload["model_state"])
    return base_flow, payload


def train_stage2(args: argparse.Namespace) -> None:
    source = load_or_generate_dataset(args)
    normalizers = build_normalizers(source)
    if not torch.isfinite(normalizers["action"].normalize(source)).all():
        raise ValueError("Normalized source dataset contains NaN/Inf; check source statistics.")

    device = torch.device(args.device)
    action_dim = source.shape[-1]
    base_flow, base_payload = load_stage1_into(args, device, action_dim)
    base_normalizers = base_payload.get("normalizers")
    if base_normalizers is not None:
        normalizers["action"].mean = base_normalizers["action_mean"].to(normalizers["action"].mean.dtype)
        normalizers["action"].std = base_normalizers["action_std"].to(normalizers["action"].std.dtype)

    warp_head = AsymmetricArmWarpHead(
        action_dim=action_dim,
        hidden_dim=args.warp_hidden_dim,
        alpha_max=args.warp_alpha_max,
        max_shift_ratio=args.warp_max_shift_ratio,
    ).to(device)

    policy = SpeedModulatedPolicy(base_flow=base_flow, warp_head=warp_head).to(device)
    policy.freeze_base_flow()

    optimizer = torch.optim.AdamW(warp_head.parameters(), lr=args.lr, weight_decay=1e-6)

    loader = DataLoader(
        TensorDataset(source),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    output = default_output(args)
    log_path = output.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    loss_kwargs = {
        "compactness_weight": args.compactness_weight,
        "content_weight": args.content_weight,
        "anchor_weight": args.anchor_weight,
        "fast_weight": args.fast_weight,
        "scale_reg_weight": args.scale_reg_weight,
    }

    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            warp_head.train()
            losses = []
            stat_acc = {}
            for (source_raw,) in loader:
                source_raw = source_raw.to(device)
                # IMPORTANT: stage 2 must operate in *raw* (un-normalized) action
                # space. Normalization converts the demonstration's zero-padding
                # to a non-zero value, which breaks active-window detection and
                # the compactness loss's "zero == idle" assumption.
                loss, log = policy.compute_warp_loss(source_raw, **loss_kwargs)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite loss detected during stage 2. Check anchor/content weights."
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(warp_head.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                for key, value in log.items():
                    stat_acc.setdefault(key, []).append(value)

            epoch_loss = float(np.mean(losses))
            stat_means = {key: float(np.mean(values)) for key, values in stat_acc.items()}
            row = {"stage": 2, "epoch": epoch, "warp_loss": epoch_loss, "lr": args.lr, **stat_means}
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            digest = " | ".join(
                [
                    f"left_shift={stat_means.get('left_shift_mean', 0):.2f}",
                    f"left_scale={stat_means.get('left_scale_mean', 0):.2f}",
                    f"right_shift={stat_means.get('right_shift_mean', 0):.2f}",
                    f"right_scale={stat_means.get('right_scale_mean', 0):.2f}",
                ]
            )
            print(f"[stage2] epoch {epoch:04d} | warp_loss={epoch_loss:.6f} | {digest}")

            payload_common = {
                "stage": 2,
                "base_flow_state": base_flow.state_dict(),
                "warp_head_state": warp_head.state_dict(),
                "epoch": epoch,
                "warp_loss": epoch_loss,
                "config": {
                    "action_dim": action_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_blocks": args.num_blocks,
                    "sample_steps": args.sample_steps,
                    "warp_hidden_dim": args.warp_hidden_dim,
                    "warp_alpha_max": args.warp_alpha_max,
                    "warp_max_shift_ratio": args.warp_max_shift_ratio,
                },
                "normalizers": {
                    "action_mean": normalizers["action"].mean.cpu(),
                    "action_std": normalizers["action"].std.cpu(),
                },
                "args": vars(args),
                "training": {
                    "uses_offline_target": False,
                    "data": "source-only; per-arm warp trained with self-supervised compactness/content/anchor losses",
                },
            }

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(output.with_name(output.stem + "_best.pt"), payload_common)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(output, payload_common)

    print(f"[stage2] Saved checkpoint: {output}")
    print(f"[stage2] Saved log: {log_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.stage == 1:
        train_stage1(args)
    else:
        train_stage2(args)


if __name__ == "__main__":
    main()
