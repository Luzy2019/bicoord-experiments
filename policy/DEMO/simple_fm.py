import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import ActionChunkDataset, DEFAULT_DATA_PATH, TimeConditionedMLP, get_device


@torch.no_grad()
def sample_flow(model: TimeConditionedMLP, shape, num_steps: int, device: torch.device) -> torch.Tensor:
    x = torch.randn(shape, device=device)
    dt = 1.0 / max(int(num_steps), 1)
    batch_size = shape[0]

    for step_idx in range(num_steps):
        t = torch.full((batch_size,), step_idx / num_steps, device=device)
        velocity = model(x, t)
        x = x + dt * velocity

    return x


def train(args):
    device = get_device(args.device)
    dataset = ActionChunkDataset(args.data, chunk_size=args.chunk_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = TimeConditionedMLP(dataset.flat_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        losses = []
        for x1 in loader:
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)
            t = torch.rand((x1.shape[0],), device=device)
            t_view = t.view(x1.shape[0], 1, 1)
            xt = (1.0 - t_view) * x0 + t_view * x1
            target_velocity = x1 - x0

            pred_velocity = model(xt, t)
            loss = F.mse_loss(pred_velocity, target_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"[FM] epoch {epoch + 1:03d} loss={sum(losses) / len(losses):.6f}")

    sample = sample_flow(model, (args.num_samples, *dataset.chunk_shape), args.flow_steps, device)
    sample = dataset.unnormalize(sample.cpu()).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))
    np.savez_compressed(args.output, samples=sample)
    print(f"saved FM samples to {args.output}, shape={sample.shape}")


def main():
    parser = argparse.ArgumentParser(description="Train a minimal flow matching policy on toy action chunks.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--flow-steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("fm_samples.npz"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
