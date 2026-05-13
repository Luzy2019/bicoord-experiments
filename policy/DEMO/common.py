from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


DEFAULT_DATA_PATH = Path(__file__).with_name("test_trajectories.npz")


class ActionChunkDataset(Dataset):
    def __init__(self, data_path: Path = DEFAULT_DATA_PATH, chunk_size: int = 10):
        data = np.load(data_path)
        actions = data["actions"].astype(np.float32)
        self.chunk_size = int(chunk_size)
        self.chunks = []

        for traj in actions:
            for start in range(0, traj.shape[0] - self.chunk_size + 1):
                self.chunks.append(traj[start : start + self.chunk_size])

        self.chunks = torch.from_numpy(np.stack(self.chunks, axis=0))
        self.mean = self.chunks.mean(dim=(0, 1), keepdim=True)
        self.std = self.chunks.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        self.normalized_chunks = (self.chunks - self.mean) / self.std

    @property
    def chunk_shape(self):
        return tuple(self.normalized_chunks.shape[1:])

    @property
    def flat_dim(self):
        chunk_size, action_dim = self.chunk_shape
        return chunk_size * action_dim

    def normalize(self, chunks: torch.Tensor) -> torch.Tensor:
        return (chunks - self.mean.to(chunks.device)) / self.std.to(chunks.device)

    def unnormalize(self, chunks: torch.Tensor) -> torch.Tensor:
        return chunks * self.std.to(chunks.device) + self.mean.to(chunks.device)

    def __len__(self):
        return self.normalized_chunks.shape[0]

    def __getitem__(self, index):
        return self.normalized_chunks[index]


class TimeConditionedMLP(nn.Module):
    def __init__(self, flat_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(flat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        t = t.reshape(batch_size, 1).to(dtype=x.dtype, device=x.device)
        out = self.net(torch.cat([x_flat, t], dim=-1))
        return out.reshape_as(x)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)
