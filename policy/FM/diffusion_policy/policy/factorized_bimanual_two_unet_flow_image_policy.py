import torch

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.factorized_bimanual_flow_unet_image_policy import FactorizedBimanualFlowUnetImagePolicy


class FactorizedBimanualTwoUnetFlowImagePolicy(FactorizedBimanualFlowUnetImagePolicy):
    """Two-arm-UNet flow-matching policy.

    The left UNet predicts both left marginal and left conditional velocities.
    The right UNet predicts both right marginal and right conditional velocities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        diffusion_step_embed_dim = kwargs.get("diffusion_step_embed_dim", 256)
        down_dims = kwargs.get("down_dims", (256, 512, 1024))
        kernel_size = kwargs.get("kernel_size", 5)
        n_groups = kwargs.get("n_groups", 8)
        cond_predict_scale = kwargs.get("cond_predict_scale", True)

        global_cond_dim = self.obs_feature_dim * self.n_obs_steps
        left_cond_dim = global_cond_dim + self.right_action_dim * 2 + 1
        right_cond_dim = global_cond_dim + self.left_action_dim * 2 + 1
        del self.factorized_model
        self.left_model = ConditionalUnet1D(
            input_dim=self.left_action_dim,
            local_cond_dim=None,
            global_cond_dim=left_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.right_model = ConditionalUnet1D(
            input_dim=self.right_action_dim,
            local_cond_dim=None,
            global_cond_dim=right_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def _arm_branch_cond(self, global_cond: torch.Tensor, other_context: torch.Tensor, cond_mask: float) -> torch.Tensor:
        batch_size = global_cond.shape[0]
        return torch.cat([global_cond, other_context, self._flag(batch_size, cond_mask, global_cond)], dim=-1)

    def _predict_factorized(self, left_state: torch.Tensor, right_state: torch.Tensor, timesteps: torch.Tensor, global_cond: torch.Tensor):
        full_state = self._combine_full(left_state, right_state)
        gates = self.factorized_gate(full_state, timesteps, global_cond, self.num_train_timesteps)
        left_context = self._context(left_state)
        right_context = self._context(right_state)
        zero_left_context = torch.zeros_like(left_context)
        zero_right_context = torch.zeros_like(right_context)

        left_marginal = self.left_model(
            left_state,
            timesteps,
            global_cond=self._arm_branch_cond(global_cond, zero_right_context, cond_mask=0.0),
        )
        right_marginal = self.right_model(
            right_state,
            timesteps,
            global_cond=self._arm_branch_cond(global_cond, zero_left_context, cond_mask=0.0),
        )
        left_cond = self.left_model(
            left_state,
            timesteps,
            global_cond=self._arm_branch_cond(global_cond, right_context, cond_mask=1.0),
        )
        right_cond = self.right_model(
            right_state,
            timesteps,
            global_cond=self._arm_branch_cond(global_cond, left_context, cond_mask=1.0),
        )

        w = gates[:, 0].view(-1, 1, 1)
        u = gates[:, 1].view(-1, 1, 1)
        left_velocity = left_marginal + w * (left_cond - left_marginal)
        right_velocity = right_marginal + u * (right_cond - right_marginal)
        gate_info = {
            "factorized_gates": gates,
            "left_marginal": left_marginal,
            "right_marginal": right_marginal,
            "left_cond": left_cond,
            "right_cond": right_cond,
        }
        self.last_gate_info = gate_info
        return left_velocity, right_velocity, gate_info
