from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeedModulationHead(nn.Module):
    """Predicts per-step execution speed alpha for an action chunk."""

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 128,
        alpha_min: float = 0.5,
        alpha_max: float = 2.0,
        init_alpha: float = 1.0,
    ):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        init_ratio = (float(init_alpha) - self.alpha_min) / max(self.alpha_max - self.alpha_min, 1e-6)
        init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
        init_bias = torch.logit(torch.tensor(init_ratio)).item()
        self.net = nn.Sequential(
            nn.Linear(action_dim + cond_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(
        self,
        signal: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: torch.Tensor = None,
        num_train_timesteps: int = 100,
    ) -> torch.Tensor:
        batch_size, horizon, _ = signal.shape
        if global_cond is None:
            global_cond = torch.zeros(batch_size, 0, device=signal.device, dtype=signal.dtype)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=signal.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(signal.device)
        timesteps = timesteps.expand(batch_size).to(signal.device).float()

        time_feat = (timesteps / max(num_train_timesteps - 1, 1)).view(batch_size, 1, 1)
        time_feat = time_feat.expand(-1, horizon, -1).to(dtype=signal.dtype)
        pos_feat = torch.linspace(0.0, 1.0, horizon, device=signal.device, dtype=signal.dtype)
        pos_feat = pos_feat.view(1, horizon, 1).expand(batch_size, -1, -1)
        cond_feat = global_cond[:, None, :].expand(-1, horizon, -1).to(dtype=signal.dtype)

        logits = self.net(torch.cat([signal, cond_feat, time_feat, pos_feat], dim=-1)).squeeze(-1)
        ratio = torch.sigmoid(logits)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * ratio


def compute_speed_alpha(
    signal: torch.Tensor,
    strength: float = 1.0,
    alpha_min: float = 0.5,
    alpha_max: float = 2.0,
    smooth_kernel: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build an execution-speed field from denoiser/vector-field magnitude.

    Larger denoiser magnitude is treated as a higher precision/risk signal, so
    it receives a smaller alpha. Smaller magnitude receives a larger alpha.
    """
    norm = signal.pow(2).mean(dim=-1).sqrt()
    if smooth_kernel > 1 and norm.shape[1] > 1:
        pad = smooth_kernel // 2
        norm = F.avg_pool1d(norm[:, None], kernel_size=smooth_kernel, stride=1, padding=pad)[:, 0]
        norm = norm[:, :signal.shape[1]]

    ref = norm.mean(dim=1, keepdim=True).clamp_min(eps)
    alpha = (ref / norm.clamp_min(eps)).pow(strength)
    alpha = alpha.clamp(min=alpha_min, max=alpha_max)
    return alpha


def normalize_risk(risk: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    risk_min = risk.amin(dim=1, keepdim=True)
    risk_max = risk.amax(dim=1, keepdim=True)
    return ((risk - risk_min) / (risk_max - risk_min).clamp_min(eps)).detach()


def compute_action_risk(actions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Estimate precision/risk from action velocity and curvature."""
    velocity = torch.zeros(actions.shape[:2], device=actions.device, dtype=actions.dtype)
    if actions.shape[1] > 1:
        velocity[:, 1:] = (actions[:, 1:] - actions[:, :-1]).pow(2).mean(dim=-1).sqrt()

    curvature = torch.zeros_like(velocity)
    if actions.shape[1] > 2:
        curvature[:, 2:] = (actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]).pow(2).mean(dim=-1).sqrt()

    return normalize_risk(velocity + 0.5 * curvature, eps=eps)


def compute_branch_mse_risk(
    conditional_pred: torch.Tensor,
    marginal_pred: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Use conditional-vs-marginal branch disagreement as coupling risk."""
    risk = (conditional_pred - marginal_pred).pow(2).mean(dim=-1)
    return normalize_risk(risk, eps=eps)


def speed_modulation_loss(
    alpha: torch.Tensor,
    actions: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
    target_weight: float = 1.0,
    smooth_weight: float = 0.1,
    fast_weight: float = 0.01,
    risk_weight: float = 0.1,
    risk: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train alpha without offline acceleration labels.

    Low-risk parts are encouraged to move faster, high-risk/curved parts are
    kept near or below normal speed, and alpha is smoothed over the chunk.
    """
    if risk is None:
        risk = compute_action_risk(actions)
    else:
        risk = risk.detach()
    target_alpha = alpha_max - risk * (alpha_max - alpha_min)
    target_loss = F.mse_loss(alpha, target_alpha)
    if alpha.shape[1] > 1:
        smooth_loss = F.mse_loss(alpha[:, 1:], alpha[:, :-1])
    else:
        smooth_loss = alpha.new_zeros(())
    fast_loss = -torch.log(alpha.clamp_min(1e-6)).mean()
    risk_loss = (risk * F.relu(alpha - 1.0).pow(2)).mean()
    total = (
        target_weight * target_loss
        + smooth_weight * smooth_loss
        + fast_weight * fast_loss
        + risk_weight * risk_loss
    )
    return total, {
        "speed_target_loss": target_loss,
        "speed_smooth_loss": smooth_loss,
        "speed_fast_loss": fast_loss,
        "speed_risk_loss": risk_loss,
        "speed_alpha_mean": alpha.mean(),
        "speed_alpha_min": alpha.min(),
        "speed_alpha_max": alpha.max(),
    }


def warp_action_sequence(actions: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Time-reparameterize an action sequence with per-step speed alpha.

    alpha > 1 advances faster along the denoised action path; alpha < 1 keeps
    denser samples around the current part of the path. Output length is
    unchanged, so existing receding-horizon code can execute it directly.
    """
    batch_size, horizon, action_dim = actions.shape
    if horizon <= 1:
        return actions

    interval_alpha = alpha[:, : horizon - 1].clamp_min(1e-6)
    zero = torch.zeros(batch_size, 1, device=actions.device, dtype=actions.dtype)
    positions = torch.cat([zero, torch.cumsum(interval_alpha, dim=1)], dim=1)
    positions = positions / positions[:, -1:].clamp_min(1e-6) * (horizon - 1)

    lower = positions.floor().long().clamp(min=0, max=horizon - 1)
    upper = (lower + 1).clamp(max=horizon - 1)
    weight = (positions - lower.to(positions.dtype)).unsqueeze(-1)

    lower_actions = actions.gather(1, lower.unsqueeze(-1).expand(-1, -1, action_dim))
    upper_actions = actions.gather(1, upper.unsqueeze(-1).expand(-1, -1, action_dim))
    return lower_actions * (1.0 - weight) + upper_actions * weight
