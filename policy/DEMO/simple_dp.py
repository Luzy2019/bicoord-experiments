import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import ActionChunkDataset, DEFAULT_DATA_PATH, TimeConditionedMLP, get_device


class SimpleDPScheduler:
    def __init__(self, num_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.num_steps = int(num_steps)
        betas = torch.linspace(beta_start, beta_end, self.num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[step].view(-1, 1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: TimeConditionedMLP, shape, device: torch.device) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for step_idx in reversed(range(self.num_steps)):
            step = torch.full((shape[0],), step_idx, device=device, dtype=torch.long)
            t = step.float() / max(self.num_steps - 1, 1)
            pred_noise = model(x, t)

            beta = self.betas[step_idx]
            alpha = self.alphas[step_idx]
            alpha_bar = self.alpha_bars[step_idx]
            mean = (x - beta / (1.0 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()

            if step_idx > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


def train(args):
    device = get_device(args.device)
    dataset = ActionChunkDataset(args.data, chunk_size=args.chunk_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = TimeConditionedMLP(dataset.flat_dim, args.hidden_dim).to(device)
    scheduler = SimpleDPScheduler(args.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        losses = []
        for x0 in loader:
            x0 = x0.to(device)
            noise = torch.randn_like(x0)
            step = torch.randint(0, scheduler.num_steps, (x0.shape[0],), device=device)
            xt = scheduler.add_noise(x0, noise, step)
            t = step.float() / max(scheduler.num_steps - 1, 1)

            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"[DP] epoch {epoch + 1:03d} loss={sum(losses) / len(losses):.6f}")

    sample = scheduler.sample(model, (args.num_samples, *dataset.chunk_shape), device)
    sample = dataset.unnormalize(sample.cpu()).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))
    import numpy as np

    np.savez_compressed(args.output, samples=sample)
    print(f"saved DP samples to {args.output}, shape={sample.shape}")


def main():
    parser = argparse.ArgumentParser(description="Train a minimal diffusion policy on toy action chunks.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("dp_samples.npz"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
