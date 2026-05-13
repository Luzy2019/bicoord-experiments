import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from common import ActionChunkDataset, DEFAULT_DATA_PATH, TimeConditionedMLP, get_device
from simple_dp import SimpleDPScheduler


def load_model(checkpoint: Path, flat_dim: int, hidden_dim: int, device: torch.device) -> TimeConditionedMLP:
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    model = TimeConditionedMLP(flat_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def record_dp_states(model: TimeConditionedMLP, shape, num_steps: int, device: torch.device, deterministic: bool = True):
    scheduler = SimpleDPScheduler(num_steps).to(device)
    x = torch.randn(shape, device=device)
    states = [x.detach().cpu()]

    for step_idx in reversed(range(scheduler.num_steps)):
        step = torch.full((shape[0],), step_idx, device=device, dtype=torch.long)
        t = step.float() / max(scheduler.num_steps - 1, 1)
        pred_noise = model(x, t)

        beta = scheduler.betas[step_idx]
        alpha = scheduler.alphas[step_idx]
        alpha_bar = scheduler.alpha_bars[step_idx]
        mean = (x - beta / (1.0 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()

        if step_idx > 0 and not deterministic:
            x = mean + beta.sqrt() * torch.randn_like(x)
        else:
            x = mean
        states.append(x.detach().cpu())

    return states


@torch.no_grad()
def record_fm_states(model: TimeConditionedMLP, shape, num_steps: int, device: torch.device):
    x = torch.randn(shape, device=device)
    dt = 1.0 / max(int(num_steps), 1)
    states = [x.detach().cpu()]

    for step_idx in range(num_steps):
        t = torch.full((shape[0],), step_idx / num_steps, device=device)
        velocity = model(x, t)
        x = x + dt * velocity
        states.append(x.detach().cpu())

    return states


def choose_target_chunk(chunks: torch.Tensor, final_chunk: torch.Tensor, target_index: int | None):
    if target_index is not None:
        index = int(target_index) % chunks.shape[0]
        return chunks[index], index

    flat_chunks = chunks.reshape(chunks.shape[0], -1)
    flat_final = final_chunk.reshape(1, -1)
    distances = torch.mean((flat_chunks - flat_final) ** 2, dim=1)
    index = int(torch.argmin(distances).item())
    return chunks[index], index


def render_compare_frame(
    current: np.ndarray,
    target: np.ndarray,
    mse_history: list[float],
    title: str,
    vmin: float,
    vmax: float,
    error_vmax: float,
) -> Image.Image:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=120)
    error = np.abs(current - target)

    current_image = axes[0, 0].imshow(current, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("current denoised chunk [10, 14]")
    axes[0, 0].set_xlabel("action dim 0..13")
    axes[0, 0].set_ylabel("chunk step 0..9")
    fig.colorbar(current_image, ax=axes[0, 0], fraction=0.046, pad=0.04)

    target_image = axes[0, 1].imshow(target, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("real target chunk [10, 14]")
    axes[0, 1].set_xlabel("action dim 0..13")
    axes[0, 1].set_ylabel("chunk step 0..9")
    fig.colorbar(target_image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    error_image = axes[1, 0].imshow(error, aspect="auto", cmap="magma", vmin=0.0, vmax=error_vmax)
    axes[1, 0].set_title("|current - real| for all 14 dims")
    axes[1, 0].set_xlabel("action dim 0..13")
    axes[1, 0].set_ylabel("chunk step 0..9")
    fig.colorbar(error_image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(mse_history, marker="o", linewidth=1.5)
    axes[1, 1].set_title("MSE to real chunk over denoising steps")
    axes[1, 1].set_xlabel("visualization frame")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, max(len(mse_history) - 1, 1))
    axes[1, 1].set_ylim(0.0, max(max(mse_history) * 1.05, 1e-6))

    mse = float(np.mean((current - target) ** 2))
    mae = float(np.mean(error))
    fig.suptitle(f"{title} | MSE={mse:.5f}, MAE={mae:.5f}")
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(rgba).convert("RGB")
    plt.close(fig)
    return image


def save_process_gif(
    states,
    dataset: ActionChunkDataset,
    output: Path,
    title_prefix: str,
    duration_ms: int,
    target_index: int | None,
):
    normalized_final = states[-1][0]
    target_normalized, resolved_target_index = choose_target_chunk(dataset.normalized_chunks, normalized_final, target_index)
    target = dataset.unnormalize(target_normalized.unsqueeze(0))[0].numpy()

    chunks = []
    for state in states:
        chunk = dataset.unnormalize(state)[0].numpy()
        chunks.append(chunk)

    stacked = np.stack(chunks, axis=0)
    limit = float(np.percentile(np.abs(np.concatenate([stacked, target[None]], axis=0)), 98))
    limit = max(limit, 1e-3)
    errors = np.abs(stacked - target[None])
    error_limit = max(float(np.percentile(errors, 98)), 1e-3)
    mse_history = [float(np.mean((chunk - target) ** 2)) for chunk in chunks]

    frames = [
        render_compare_frame(
            chunk,
            target,
            mse_history[: idx + 1],
            f"{title_prefix} step {idx}/{len(chunks) - 1}, target chunk #{resolved_target_index}",
            -limit,
            limit,
            error_limit,
        )
        for idx, chunk in enumerate(chunks)
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"saved {output}")
    print(f"{title_prefix} target chunk index: {resolved_target_index}")
    print(f"{title_prefix} final MSE: {mse_history[-1]:.6f}, final MAE: {float(np.mean(errors[-1])):.6f}")


def main():
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Visualize DP denoising and FM flow process as GIFs.")
    parser.add_argument("--method", choices=["dp", "fm", "both"], default="both")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dp-checkpoint", type=Path, default=default_dir / "dp_samples.pt")
    parser.add_argument("--fm-checkpoint", type=Path, default=default_dir / "fm_samples.pt")
    parser.add_argument("--dp-steps", type=int, default=10)
    parser.add_argument("--fm-steps", type=int, default=50)
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--stochastic-dp", action="store_true")
    parser.add_argument("--duration-ms", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=default_dir / "visualizations")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    dataset = ActionChunkDataset(args.data, chunk_size=args.chunk_size)
    shape = (1, *dataset.chunk_shape)

    if args.method in ("dp", "both"):
        dp_model = load_model(args.dp_checkpoint, dataset.flat_dim, args.hidden_dim, device)
        dp_states = record_dp_states(dp_model, shape, args.dp_steps, device, deterministic=not args.stochastic_dp)
        save_process_gif(
            dp_states,
            dataset,
            args.output_dir / "dp_denoising_compare.gif",
            "DP denoising",
            args.duration_ms,
            args.target_index,
        )

    if args.method in ("fm", "both"):
        fm_model = load_model(args.fm_checkpoint, dataset.flat_dim, args.hidden_dim, device)
        fm_states = record_fm_states(fm_model, shape, args.fm_steps, device)
        save_process_gif(
            fm_states,
            dataset,
            args.output_dir / "fm_flow_compare.gif",
            "FM flow",
            args.duration_ms,
            args.target_index,
        )


if __name__ == "__main__":
    main()
