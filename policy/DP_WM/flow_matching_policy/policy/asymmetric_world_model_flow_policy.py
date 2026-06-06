from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching_policy.common.pytorch_util import dict_apply
from flow_matching_policy.policy.factorized_bimanual_diffusion_unet_image_policy import (
    FactorizedBimanualDiffusionUnetImagePolicy,
)


class LatentActionWorldModel(nn.Module):
    """LeWorldModel-style latent dynamics head.

    The head learns z_{t+1} from z_t and a_t in a compact latent space. A
    Gaussian latent regularizer keeps the embedding from collapsing while still
    allowing the shared observation encoder to receive dynamics supervision.
    """

    def __init__(
        self,
        obs_feature_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        detach_target: bool = True,
        predict_delta: bool = True,
    ):
        super().__init__()
        self.detach_target = bool(detach_target)
        self.predict_delta = bool(predict_delta)
        self.projector = nn.Sequential(
            nn.LayerNorm(obs_feature_dim),
            nn.Linear(obs_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.dynamics = nn.Sequential(
            nn.LayerNorm(latent_dim + action_dim),
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, obs_features: torch.Tensor) -> torch.Tensor:
        return self.projector(obs_features)

    def predict_next(self, z_now: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        update = self.dynamics(torch.cat([z_now, action], dim=-1))
        if self.predict_delta:
            return z_now + update
        return update

    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        gaussian_weight: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if obs_features.shape[1] < 2:
            zero = obs_features.new_zeros(())
            return zero, {
                "world_pred_loss": zero,
                "world_gaussian_loss": zero,
                "world_latent_std": zero,
            }

        z = self.encode(obs_features)
        z_now = z[:, :-1]
        z_next = z[:, 1:]
        transition_actions = actions[:, : z_now.shape[1]]
        pred_next = self.predict_next(z_now, transition_actions)
        target_next = z_next.detach() if self.detach_target else z_next

        pred_loss = F.mse_loss(pred_next, target_next)
        flat_z = z.reshape(-1, z.shape[-1])
        latent_mean = flat_z.mean(dim=0)
        latent_std = flat_z.std(dim=0, unbiased=False)
        gaussian_loss = latent_mean.pow(2).mean() + (latent_std - 1.0).pow(2).mean()
        total = pred_loss + float(gaussian_weight) * gaussian_loss
        return total, {
            "world_pred_loss": pred_loss,
            "world_gaussian_loss": gaussian_loss,
            "world_latent_std": latent_std.mean(),
        }


class CouplingEnergyHead(nn.Module):
    """Scalar energy for action-level coupling feasibility.

    This is a lightweight EBM-style residual: expert action chunks are trained
    to have lower energy than corrupted left/right pairings. At inference it can
    rerank multiple flow samples without changing the flow sampler itself.
    """

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim * 2 + cond_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        full_action: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = full_action.shape[0]
        if global_cond is None:
            global_cond = torch.zeros(
                batch_size,
                0,
                device=full_action.device,
                dtype=full_action.dtype,
            )
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=full_action.dtype, device=full_action.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(full_action.device)
        timesteps = timesteps.expand(batch_size).to(device=full_action.device, dtype=full_action.dtype)
        time_feat = timesteps.view(batch_size, 1)
        pooled = torch.cat(
            [
                full_action.mean(dim=1),
                full_action.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        return self.net(torch.cat([pooled, global_cond, time_feat], dim=-1)).squeeze(-1)


class AsymmetricWorldModelFlowPolicy(FactorizedBimanualDiffusionUnetImagePolicy):
    """Factorized bimanual Flow Matching with latent world-model supervision.

    This policy implements the first executable version of the asymmetric
    coupling decomposition:

    local arm field + gated cross-arm residual + optional coupling energy.

    The inherited flow branches provide p(a_L | O), p(a_R | O),
    p(a_L | O, a_R), and p(a_R | O, a_L). This subclass adds:
    - LeWM-style latent next-embedding prediction.
    - Orthogonal vector-field regularization for left/right residuals.
    - Gate sparsity, matching the Occam/ABC selection view.
    - EBM-style contrastive coupling energy and optional sample reranking.
    """

    def __init__(
        self,
        *args,
        world_model_enabled: bool = True,
        world_model_latent_dim: int = 128,
        world_model_hidden_dim: int = 256,
        world_model_loss_weight: float = 0.05,
        world_model_gaussian_weight: float = 0.01,
        world_model_detach_target: bool = True,
        orthogonal_loss_weight: float = 0.01,
        gate_sparsity_loss_weight: float = 0.001,
        residual_balance_loss_weight: float = 0.001,
        energy_loss_weight: float = 0.01,
        energy_hidden_dim: int = 256,
        energy_margin: float = 1.0,
        energy_rerank_samples: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.world_model_enabled = bool(world_model_enabled)
        self.world_model_loss_weight = float(world_model_loss_weight)
        self.world_model_gaussian_weight = float(world_model_gaussian_weight)
        self.orthogonal_loss_weight = float(orthogonal_loss_weight)
        self.gate_sparsity_loss_weight = float(gate_sparsity_loss_weight)
        self.residual_balance_loss_weight = float(residual_balance_loss_weight)
        self.energy_loss_weight = float(energy_loss_weight)
        self.energy_margin = float(energy_margin)
        self.energy_rerank_samples = int(energy_rerank_samples)

        if self.world_model_enabled:
            self.latent_world_model = LatentActionWorldModel(
                obs_feature_dim=self.obs_feature_dim,
                action_dim=self.action_dim,
                latent_dim=int(world_model_latent_dim),
                hidden_dim=int(world_model_hidden_dim),
                detach_target=bool(world_model_detach_target),
            )
        else:
            self.latent_world_model = None

        self.coupling_energy = CouplingEnergyHead(
            action_dim=self.action_dim,
            cond_dim=self.obs_feature_dim * self.n_obs_steps,
            hidden_dim=int(energy_hidden_dim),
        )

    def _encode_obs_sequence(self, normalized_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        value = next(iter(normalized_obs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(
            normalized_obs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        features = self.obs_encoder(this_nobs)
        return features.reshape(batch_size, self.n_obs_steps, -1)

    @staticmethod
    def _orthogonal_loss(left_field: torch.Tensor, right_field: torch.Tensor) -> torch.Tensor:
        if left_field.shape[-1] != right_field.shape[-1]:
            return left_field.new_zeros(())
        left_unit = F.normalize(left_field, dim=-1)
        right_unit = F.normalize(right_field, dim=-1)
        return (left_unit * right_unit).sum(dim=-1).pow(2).mean()

    @staticmethod
    def _residual_balance_loss(left_residual: torch.Tensor, right_residual: torch.Tensor) -> torch.Tensor:
        left_energy = left_residual.pow(2).mean(dim=(-1, -2))
        right_energy = right_residual.pow(2).mean(dim=(-1, -2))
        return F.mse_loss(left_energy, right_energy)

    def _energy_contrastive_loss(
        self,
        normalized_actions: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = normalized_actions.shape[0]
        final_t = torch.ones(batch_size, device=normalized_actions.device, dtype=normalized_actions.dtype)
        positive_energy = self.coupling_energy(normalized_actions, final_t, global_cond)

        if batch_size > 1:
            perm = torch.randperm(batch_size, device=normalized_actions.device)
            negative = normalized_actions.clone()
            negative[:, :, self.right_slice] = normalized_actions[perm, :, self.right_slice]
        else:
            negative = normalized_actions + torch.randn_like(normalized_actions) * 0.25
        negative_energy = self.coupling_energy(negative, final_t, global_cond)
        ranking_loss = F.softplus(positive_energy - negative_energy + self.energy_margin).mean()
        return ranking_loss, {
            "coupling_energy_positive": positive_energy.mean(),
            "coupling_energy_negative": negative_energy.mean(),
        }

    def compute_loss(self, batch):
        base_loss = super().compute_loss(batch)
        total_loss = base_loss
        extra_logs = {}

        gate_info = self.last_gate_info
        if gate_info is not None:
            gates = gate_info["factorized_gates"]
            left_residual = gate_info["left_cond"] - gate_info["left_marginal"]
            right_residual = gate_info["right_cond"] - gate_info["right_marginal"]
            left_final = gate_info["left_marginal"] + gates[:, 0].view(-1, 1, 1) * left_residual
            right_final = gate_info["right_marginal"] + gates[:, 1].view(-1, 1, 1) * right_residual

            gate_sparsity_loss = gates.mean()
            orthogonal_loss = self._orthogonal_loss(left_final, right_final)
            residual_balance_loss = self._residual_balance_loss(left_residual, right_residual)
            coupling_residual_energy = 0.5 * (
                left_residual.pow(2).mean() + right_residual.pow(2).mean()
            )

            total_loss = (
                total_loss
                + self.gate_sparsity_loss_weight * gate_sparsity_loss
                + self.orthogonal_loss_weight * orthogonal_loss
                + self.residual_balance_loss_weight * residual_balance_loss
            )
            extra_logs.update(
                {
                    "gate_sparsity_loss": gate_sparsity_loss,
                    "orthogonal_loss": orthogonal_loss,
                    "residual_balance_loss": residual_balance_loss,
                    "coupling_residual_energy": coupling_residual_energy,
                }
            )

        normalized_obs = self.normalizer.normalize(batch["obs"])
        normalized_actions = self.normalizer["action"].normalize(batch["action"])
        obs_features = self._encode_obs_sequence(normalized_obs)
        global_cond = obs_features.reshape(obs_features.shape[0], -1)

        if self.latent_world_model is not None and self.world_model_loss_weight > 0:
            world_loss, world_logs = self.latent_world_model.compute_loss(
                obs_features=obs_features,
                actions=normalized_actions,
                gaussian_weight=self.world_model_gaussian_weight,
            )
            total_loss = total_loss + self.world_model_loss_weight * world_loss
            extra_logs["world_model_loss"] = world_loss
            extra_logs.update(world_logs)

        if self.energy_loss_weight > 0:
            energy_loss, energy_logs = self._energy_contrastive_loss(normalized_actions, global_cond)
            total_loss = total_loss + self.energy_loss_weight * energy_loss
            extra_logs["energy_contrastive_loss"] = energy_loss
            extra_logs.update(energy_logs)

        self.last_loss_dict.update(
            {key: float(value.detach().cpu()) for key, value in extra_logs.items()}
        )
        return total_loss

    def _select_lowest_energy_candidate(
        self,
        candidates: torch.Tensor,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        # candidates: K, B, T, D; energies: K, B
        best = energies.argmin(dim=0)
        batch_index = torch.arange(candidates.shape[1], device=candidates.device)
        return candidates[best, batch_index]

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.energy_rerank_samples <= 1:
            return super().predict_action(obs_dict)

        assert "past_action" not in obs_dict
        value = next(iter(obs_dict.values()))
        batch_size = value.shape[0]
        global_cond = self._encode_obs(obs_dict)

        candidates = []
        energies = []
        final_t = torch.ones(batch_size, dtype=global_cond.dtype, device=global_cond.device)
        for sample_idx in range(self.energy_rerank_samples):
            left_sample, right_sample = self._conditional_sample(
                batch_size,
                global_cond,
                collect_debug=(sample_idx == 0),
            )
            normalized_full = self._combine_full(left_sample, right_sample)
            candidates.append(normalized_full)
            energies.append(self.coupling_energy(normalized_full, final_t, global_cond))

        candidate_tensor = torch.stack(candidates, dim=0)
        energy_tensor = torch.stack(energies, dim=0)
        normalized_best = self._select_lowest_energy_candidate(candidate_tensor, energy_tensor)
        action_pred = self.normalizer["action"].unnormalize(normalized_best)
        self.last_action_pred_raw = action_pred.detach()

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        result = {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
            "candidate_energy": energy_tensor.detach(),
        }
        if self.last_gate_info is not None:
            result["factorized_gates"] = self.last_gate_info["factorized_gates"]
        return result
