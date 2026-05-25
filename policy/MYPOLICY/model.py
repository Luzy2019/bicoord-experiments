"""MYPOLICY model definitions.

Design (matches `policy/FM` reference + the two reference docs):

    Stage 1 (base flow):  v_theta(x_t, t) trained with rectified flow on
                          source-only trajectories. Learns the task manifold.

    Stage 2 (per-arm speed/warp head): an asymmetric per-arm time reparameterization
                          alpha_L, alpha_R (encoded as per-arm (shift, scale))
                          trained self-supervised on source only. Encourages each
                          arm's active window to land at t=0 (i.e. parallel
                          execution) without ever consuming a precomputed
                          "parallel target" label.

At inference, sampling from the base flow gives a source-like (sequential) trajectory.
Applying the per-arm warp turns it into a parallel one. No `target` data is touched
during training (only during evaluation/visualization).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LEFT_DIM = 7
RIGHT_DIM = 7
ACTION_DIM = LEFT_DIM + RIGHT_DIM


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean.to(value.device)) / self.std.to(value.device)

    def unnormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std.to(value.device) + self.mean.to(value.device)


def build_normalizers(source: torch.Tensor) -> Dict[str, Normalizer]:
    action_mean = source.mean(dim=(0, 1), keepdim=True)
    action_std = source.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    return {"action": Normalizer(action_mean, action_std)}


# ---------------------------------------------------------------------------
# Base flow network (Stage 1, unconditional rectified flow on source)
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * -(torch.log(torch.tensor(10000.0, device=time.device, dtype=time.dtype)) / max(half - 1, 1))
        )
        emb = time[:, None] * freq[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, 1))
        return emb


class ConvResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TrajectoryFlowNet(nn.Module):
    """Unconditional 1D conv vector field for bimanual action trajectories.

    Input  : x_t in R^{B x H x A}, flow-time t in R^{B}.
    Output : v(x_t, t) in R^{B x H x A}.
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input = nn.Conv1d(action_dim + 1, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([ConvResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
            nn.Conv1d(hidden_dim, action_dim, kernel_size=1),
        )

    def forward(self, xt: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = xt.shape
        time_channel = time.view(batch_size, 1, 1).expand(-1, horizon, 1)
        x = torch.cat([xt, time_channel], dim=-1).transpose(1, 2)
        h = self.input(x)
        time_feature = self.time_embed(time).unsqueeze(-1)
        h = h + time_feature
        for block in self.blocks:
            h = block(h)
        return self.output(h).transpose(1, 2)


class TrajectoryFlowMatchingPolicy(nn.Module):
    """Stage-1 base flow: rectified flow trained on source-only trajectories.

    The network learns to integrate Gaussian noise into the source distribution
    (sequential bimanual demonstrations). It does NOT see any "parallel target"
    label during training -- that comes later from the Stage-2 warp head.
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.net = TrajectoryFlowNet(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            time_embed_dim=time_embed_dim,
        )

    def compute_loss(self, source: torch.Tensor) -> torch.Tensor:
        """Standard rectified-flow MSE between predicted v and (source - noise)."""
        batch_size = source.shape[0]
        noise = torch.randn_like(source)
        t = torch.rand(batch_size, device=source.device, dtype=source.dtype)
        t_view = t.view(batch_size, *([1] * (source.ndim - 1)))
        xt = (1.0 - t_view) * noise + t_view * source
        velocity = source - noise
        pred = self.net(xt, t)
        return F.mse_loss(pred, velocity)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        num_steps: int = 100,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        x = torch.randn(
            batch_size,
            horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        steps = max(int(num_steps), 1)
        dt = 1.0 / steps
        for step_idx in range(steps):
            t = torch.full(
                (batch_size,),
                step_idx / steps,
                device=device,
                dtype=dtype,
            )
            x = x + dt * self.net(x, t)
        return x


# ---------------------------------------------------------------------------
# Stage 2: per-arm time reparameterization (asymmetric speed field)
# ---------------------------------------------------------------------------


def detect_active_start(arm_actions: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Return per-batch index of the first non-zero step of an arm slice [B, H, D].

    Computed online from the source batch. No precomputed parallel target labels
    are involved; this is just the natural boundary of the demonstration's idle
    prefix.
    """
    active = torch.linalg.norm(arm_actions, dim=-1) > eps
    has_any = active.any(dim=1)
    first = active.float().argmax(dim=1)
    return torch.where(has_any, first, torch.zeros_like(first))


def _affine_warp(
    arm_actions: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Per-arm affine time reparameterization.

    output[i] = arm_actions[shift + i * scale]
    Positions beyond H-1 are clamped to 0 (zero-padding past the warped sequence),
    so the warped output retains the same horizon as the original.
    """
    batch_size, horizon, dim = arm_actions.shape
    if horizon <= 1:
        return arm_actions
    grid = torch.arange(horizon, device=arm_actions.device, dtype=arm_actions.dtype).view(1, horizon)
    pos = shift.view(batch_size, 1) + grid * scale.view(batch_size, 1)
    valid = (pos <= (horizon - 1)).to(arm_actions.dtype).unsqueeze(-1)
    pos_clamped = pos.clamp(0.0, float(horizon - 1))
    lower = pos_clamped.floor().long().clamp(0, horizon - 1)
    upper = (lower + 1).clamp(0, horizon - 1)
    weight = (pos_clamped - lower.to(pos_clamped.dtype)).unsqueeze(-1)
    lower_a = arm_actions.gather(1, lower.unsqueeze(-1).expand(-1, -1, dim))
    upper_a = arm_actions.gather(1, upper.unsqueeze(-1).expand(-1, -1, dim))
    out = lower_a * (1.0 - weight) + upper_a * weight
    return out * valid


def per_arm_affine_warp(
    actions: torch.Tensor,
    shift_L: torch.Tensor,
    scale_L: torch.Tensor,
    shift_R: torch.Tensor,
    scale_R: torch.Tensor,
    left_dim: int = LEFT_DIM,
    right_dim: int = RIGHT_DIM,
) -> torch.Tensor:
    out = torch.zeros_like(actions)
    out[..., :left_dim] = _affine_warp(actions[..., :left_dim], shift_L, scale_L)
    out[..., left_dim : left_dim + right_dim] = _affine_warp(
        actions[..., left_dim : left_dim + right_dim], shift_R, scale_R
    )
    return out


class AsymmetricArmWarpHead(nn.Module):
    """Predicts per-arm (shift, scale) for time reparameterization.

    This is the asymmetric per-arm analogue of `SpeedModulationHead` in
    `policy/FM/.../speed_modulation.py`. Each arm gets its OWN (shift, scale)
    so the left and right arms can be re-timed independently -- the
    "asymmetric per-arm decoupling" from the reference document.

    Output bounds (via sigmoid):
        shift in [0, horizon * max_shift_ratio]
        scale in [1.0, alpha_max]   (scale >= 1 -> only compress, never stretch)

    Identity-warp initialization (sigmoid biased to near 0) keeps Stage 2
    starting close to "do nothing", so the base flow's behavior is preserved
    until the self-supervised losses guide the head.
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 128,
        max_shift_ratio: float = 0.6,
        alpha_max: float = 3.0,
        shift_init_bias: float = 0.0,
        scale_init_bias: float = -3.0,
    ):
        super().__init__()
        self.max_shift_ratio = float(max_shift_ratio)
        self.alpha_max = float(alpha_max)
        # The encoder injects a positional channel so it can attend to *when*
        # each arm becomes active (a per-sample feature). Without positional
        # info a 1D conv + global pool only sees aggregate statistics that
        # are nearly identical across samples, and the head collapses to the
        # batch-mean prediction.
        self.encoder = nn.Sequential(
            nn.Conv1d(action_dim + 1, hidden_dim, kernel_size=5, padding=2),
            nn.Mish(),
            nn.GroupNorm(8, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Mish(),
            nn.GroupNorm(8, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Mish(),
        )
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.head[-1].weight)
        bias = torch.tensor([
            float(shift_init_bias),
            float(scale_init_bias),
            float(shift_init_bias),
            float(scale_init_bias),
        ])
        with torch.no_grad():
            self.head[-1].bias.copy_(bias)

    def forward(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, horizon, _ = actions.shape
        pos = torch.linspace(0.0, 1.0, horizon, device=actions.device, dtype=actions.dtype)
        pos = pos.view(1, horizon, 1).expand(batch_size, -1, -1)
        feat_in = torch.cat([actions, pos], dim=-1)
        feat_seq = self.encoder(feat_in.transpose(1, 2))
        feat = torch.cat(
            [self.pool_avg(feat_seq).squeeze(-1), self.pool_max(feat_seq).squeeze(-1)],
            dim=-1,
        )
        logits = self.head(feat)
        ratios = torch.sigmoid(logits)
        max_shift = float(horizon - 1) * self.max_shift_ratio
        shift_L = ratios[:, 0] * max_shift
        scale_L = 1.0 + ratios[:, 1] * (self.alpha_max - 1.0)
        shift_R = ratios[:, 2] * max_shift
        scale_R = 1.0 + ratios[:, 3] * (self.alpha_max - 1.0)
        return {
            "shift_L": shift_L,
            "scale_L": scale_L,
            "shift_R": shift_R,
            "scale_R": scale_R,
        }


def asymmetric_warp_loss(
    warped: torch.Tensor,
    source: torch.Tensor,
    params: Dict[str, torch.Tensor],
    left_dim: int = LEFT_DIM,
    right_dim: int = RIGHT_DIM,
    compactness_weight: float = 1.0,
    content_weight: float = 0.05,
    anchor_weight: float = 5.0,
    fast_weight: float = 0.0,
    scale_reg_weight: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Self-supervised loss for the asymmetric per-arm warp head.

    Components (per arm, source-only -- no precomputed target trajectories):
      * compactness:  position-weighted L1 norm of the warped arm.
                      Minimized when the active content is concentrated near t=0,
                      i.e. both arms become parallel from the start.
      * content   :   the warped arm should preserve total action mass
                      (scale * sum(|warped|) ~ sum(|source|)), so the head
                      cannot "cheat" by deleting the active content.
                      Computed as a *relative* error so its magnitude is
                      independent of action norm scale.
      * anchor    :   the predicted shift should match the demonstration's
                      online-detected active-window start. This is a self-
                      supervised supervision computed on the fly from the
                      source batch -- it stabilizes Stage-2 training.
      * fast      :   -log(scale), optional prior encouraging scale > 1 (=>
                      time compression, faster execution). Default weight is 0
                      because pushing scale up tends to find a "compress-only"
                      local minimum that does NOT shift the active window;
                      enable it explicitly if you want extra speedup beyond
                      parallel alignment.
      * scale_reg :   (scale - 1)^2 trust region anchoring scale at 1, so the
                      head must obtain compactness via *shift* (parallel
                      alignment) rather than by uniformly compressing
                      everything (which would also collapse the left arm).
    """
    horizon = source.shape[1]
    pos_weight = torch.arange(horizon, device=source.device, dtype=source.dtype) / max(horizon - 1, 1)
    pos_weight = pos_weight.view(1, horizon)

    total = source.new_zeros(())
    log: Dict[str, float] = {}

    arms = (
        ("left", slice(0, left_dim), params["shift_L"], params["scale_L"]),
        (
            "right",
            slice(left_dim, left_dim + right_dim),
            params["shift_R"],
            params["scale_R"],
        ),
    )

    for name, slc, shift, scale in arms:
        warped_arm = warped[..., slc]
        source_arm = source[..., slc]
        norm_warped = torch.linalg.norm(warped_arm, dim=-1)
        norm_source = torch.linalg.norm(source_arm, dim=-1)

        total_warped = norm_warped.sum(dim=1).clamp_min(eps)
        compactness = (pos_weight * norm_warped).sum(dim=1) / total_warped
        compactness_loss = compactness.mean()

        total_source = norm_source.sum(dim=1).clamp_min(eps)
        # Relative content-mass error so the scale doesn't dominate the loss
        content_loss = ((total_warped * scale - total_source) / total_source).pow(2).mean()

        online_start = detect_active_start(source_arm).to(shift.dtype).detach()
        anchor_loss = ((shift - online_start) / float(horizon)).abs().mean()

        fast_loss = -torch.log(scale.clamp_min(eps)).mean()
        scale_reg_loss = (scale - 1.0).pow(2).mean()

        arm_total = (
            compactness_weight * compactness_loss
            + content_weight * content_loss
            + anchor_weight * anchor_loss
            + fast_weight * fast_loss
            + scale_reg_weight * scale_reg_loss
        )
        total = total + arm_total

        log[f"{name}_compactness"] = float(compactness_loss.detach().cpu())
        log[f"{name}_content"] = float(content_loss.detach().cpu())
        log[f"{name}_anchor"] = float(anchor_loss.detach().cpu())
        log[f"{name}_fast"] = float(fast_loss.detach().cpu())
        log[f"{name}_scale_reg"] = float(scale_reg_loss.detach().cpu())
        log[f"{name}_shift_mean"] = float(shift.mean().detach().cpu())
        log[f"{name}_scale_mean"] = float(scale.mean().detach().cpu())

    return total, log


# ---------------------------------------------------------------------------
# Combined policy (Stage 2 / inference)
# ---------------------------------------------------------------------------


class SpeedModulatedPolicy(nn.Module):
    """Base flow + asymmetric per-arm warp head."""

    def __init__(
        self,
        base_flow: TrajectoryFlowMatchingPolicy,
        warp_head: AsymmetricArmWarpHead,
        left_dim: int = LEFT_DIM,
        right_dim: int = RIGHT_DIM,
    ):
        super().__init__()
        self.base_flow = base_flow
        self.warp_head = warp_head
        self.left_dim = left_dim
        self.right_dim = right_dim

    def compute_warp_loss(
        self,
        source: torch.Tensor,
        **loss_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        params = self.warp_head(source)
        warped = per_arm_affine_warp(
            source,
            params["shift_L"],
            params["scale_L"],
            params["shift_R"],
            params["scale_R"],
            left_dim=self.left_dim,
            right_dim=self.right_dim,
        )
        return asymmetric_warp_loss(
            warped=warped,
            source=source,
            params=params,
            left_dim=self.left_dim,
            right_dim=self.right_dim,
            **loss_kwargs,
        )

    def freeze_base_flow(self) -> None:
        for parameter in self.base_flow.parameters():
            parameter.requires_grad = False
        self.base_flow.eval()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        num_steps: int = 100,
        generator: Optional[torch.Generator] = None,
        unnormalize=None,
    ) -> Dict[str, torch.Tensor]:
        """Sample raw action chunks then apply the per-arm warp.

        If `unnormalize` is provided, it is applied to the base-flow output
        before the warp so that the warp head sees actions in raw (zero-
        padded) space -- which is what it was trained on. The returned `raw`
        and `warped` tensors are in raw space.
        """
        raw_norm = self.base_flow.sample(
            batch_size=batch_size,
            horizon=horizon,
            device=device,
            dtype=dtype,
            num_steps=num_steps,
            generator=generator,
        )
        raw = unnormalize(raw_norm) if unnormalize is not None else raw_norm
        params = self.warp_head(raw)
        warped = per_arm_affine_warp(
            raw,
            params["shift_L"],
            params["scale_L"],
            params["shift_R"],
            params["scale_R"],
            left_dim=self.left_dim,
            right_dim=self.right_dim,
        )
        return {"raw": raw, "warped": warped, **params}

    @torch.no_grad()
    def warp_only(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        params = self.warp_head(actions)
        warped = per_arm_affine_warp(
            actions,
            params["shift_L"],
            params["scale_L"],
            params["shift_R"],
            params["scale_R"],
            left_dim=self.left_dim,
            right_dim=self.right_dim,
        )
        return {"raw": actions, "warped": warped, **params}
