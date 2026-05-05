import torch
from torch import nn
from torch.nn import functional as F


def _validate_stride(stride, name):
    if stride < 1:
        raise ValueError(f"{name} must be >= 1, got {stride}")


def _build_update_mask(horizon, stride, batch_size, device):
    _validate_stride(stride, "stride")
    step_ids = torch.arange(horizon, device=device)
    mask = (step_ids % stride == 0).float().view(1, horizon, 1)
    return mask.expand(batch_size, -1, -1)


def _build_phase_index(horizon, stride, batch_size, device):
    _validate_stride(stride, "stride")
    step_ids = torch.arange(horizon, device=device)
    phase = (step_ids % stride).long().view(1, horizon)
    return phase.expand(batch_size, -1)


def _slice_to_tuple(index_slice):
    return (index_slice.start, index_slice.stop)


class FrequencyProjector(nn.Module):
    """
    Projects a unified action chunk onto asymmetric arm update schedules.

    The tensor shape stays [B, H, D]. Low-frequency arms are represented by
    holding their latest update value on non-update steps.
    """

    def __init__(
        self,
        left_slice,
        right_slice,
        left_stride=1,
        right_stride=1,
        mode="hold",
    ):
        super().__init__()
        if mode not in ("hold", "none"):
            raise ValueError(f"Unsupported frequency projector mode: {mode}")
        _validate_stride(left_stride, "left_stride")
        _validate_stride(right_stride, "right_stride")

        self.left_slice = left_slice
        self.right_slice = right_slice
        self.left_stride = left_stride
        self.right_stride = right_stride
        self.mode = mode

    @staticmethod
    def _hold_project(arm_actions, stride):
        if stride <= 1:
            return arm_actions

        _, horizon, _ = arm_actions.shape
        projected = arm_actions.clone()
        last_action = arm_actions[:, 0]
        for step in range(horizon):
            if step % stride == 0:
                last_action = arm_actions[:, step]
            else:
                projected[:, step] = last_action
        return projected

    def forward(self, actions, left_stride=None, right_stride=None):
        batch_size, horizon, _ = actions.shape
        device = actions.device
        left_stride = self.left_stride if left_stride is None else left_stride
        right_stride = self.right_stride if right_stride is None else right_stride
        _validate_stride(left_stride, "left_stride")
        _validate_stride(right_stride, "right_stride")

        projected = actions.clone()
        if self.mode == "hold":
            projected[:, :, self.left_slice] = self._hold_project(
                projected[:, :, self.left_slice], left_stride
            )
            projected[:, :, self.right_slice] = self._hold_project(
                projected[:, :, self.right_slice], right_stride
            )

        info = {
            "left_update_mask": _build_update_mask(horizon, left_stride, batch_size, device),
            "right_update_mask": _build_update_mask(horizon, right_stride, batch_size, device),
            "left_phase": _build_phase_index(horizon, left_stride, batch_size, device),
            "right_phase": _build_phase_index(horizon, right_stride, batch_size, device),
            "left_slice": _slice_to_tuple(self.left_slice),
            "right_slice": _slice_to_tuple(self.right_slice),
            "left_stride": left_stride,
            "right_stride": right_stride,
        }
        return projected, info


class AsymFreqResidualAdapter(nn.Module):
    """
    Frequency-aware residual head for bimanual ACT action chunks.

    final = freq_action + residual_mask * gate * delta
    """

    def __init__(
        self,
        action_dim,
        horizon,
        left_slice,
        right_slice,
        left_stride=1,
        right_stride=1,
        cond_dim=0,
        hidden_dim=256,
        max_stride=16,
        residual_scale=0.1,
        projector_mode="hold",
        mask_residual_non_update=True,
    ):
        super().__init__()
        max_stride = max(max_stride, left_stride, right_stride)

        self.action_dim = action_dim
        self.horizon = horizon
        self.cond_dim = cond_dim
        self.left_slice = left_slice
        self.right_slice = right_slice
        self.residual_scale = residual_scale
        self.mask_residual_non_update = mask_residual_non_update

        self.projector = FrequencyProjector(
            left_slice=left_slice,
            right_slice=right_slice,
            left_stride=left_stride,
            right_stride=right_stride,
            mode=projector_mode,
        )

        self.left_phase_emb = nn.Embedding(max_stride, hidden_dim)
        self.right_phase_emb = nn.Embedding(max_stride, hidden_dim)

        input_dim = action_dim + hidden_dim * 2 + 2 + cond_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, action_dim)
        self.gate_head = nn.Linear(hidden_dim, action_dim)

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -4.0)

    def _residual_mask(self, freq_info):
        batch_size = freq_info["left_update_mask"].shape[0]
        horizon = freq_info["left_update_mask"].shape[1]
        device = freq_info["left_update_mask"].device
        mask = torch.ones(batch_size, horizon, self.action_dim, device=device)
        if self.mask_residual_non_update:
            mask[:, :, self.left_slice] = freq_info["left_update_mask"]
            mask[:, :, self.right_slice] = freq_info["right_update_mask"]
        return mask

    def forward(self, action_chunk, cond=None, left_stride=None, right_stride=None):
        batch_size, horizon, action_dim = action_chunk.shape
        if horizon != self.horizon:
            raise ValueError(f"Expected horizon {self.horizon}, got {horizon}")
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim {self.action_dim}, got {action_dim}")
        if self.cond_dim > 0 and cond is None:
            raise ValueError("cond must be provided when cond_dim > 0")

        freq_action, freq_info = self.projector(
            action_chunk,
            left_stride=left_stride,
            right_stride=right_stride,
        )
        left_phase_emb = self.left_phase_emb(freq_info["left_phase"])
        right_phase_emb = self.right_phase_emb(freq_info["right_phase"])
        phase_feat = torch.cat([left_phase_emb, right_phase_emb], dim=-1)
        mask_feat = torch.cat(
            [freq_info["left_update_mask"], freq_info["right_update_mask"]],
            dim=-1,
        )

        features = [freq_action, phase_feat, mask_feat]
        if self.cond_dim > 0:
            features.append(cond[:, None, :].expand(batch_size, horizon, -1))

        hidden = self.trunk(torch.cat(features, dim=-1))
        delta = self.delta_head(hidden) * self.residual_scale
        gate = torch.sigmoid(self.gate_head(hidden))
        residual_mask = self._residual_mask(freq_info)
        final_action = freq_action + residual_mask * gate * delta

        info = {
            "base": action_chunk,
            "freq": freq_action,
            "delta": delta,
            "gate": gate,
            "residual_mask": residual_mask,
            "frequency": freq_info,
        }
        return final_action, info


class LearnedAsymFreqResidualAdapter(nn.Module):
    """
    Learns per-arm update gates from synchronous demonstrations.

    Instead of receiving fixed left/right frequencies, this module predicts
    update probabilities for both arms at every chunk step. Low update
    probability softly holds the previous action, while high probability lets
    the current base action pass through.
    """

    def __init__(
        self,
        action_dim,
        horizon,
        left_slice,
        right_slice,
        cond_dim=0,
        hidden_dim=256,
        residual_scale=0.1,
        hard_inference=True,
        update_threshold=0.5,
        num_roles=4,
        role_emb_dim=16,
        use_role_condition=True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.cond_dim = cond_dim
        self.left_slice = left_slice
        self.right_slice = right_slice
        self.residual_scale = residual_scale
        self.hard_inference = hard_inference
        self.update_threshold = update_threshold
        self.num_roles = num_roles
        self.use_role_condition = use_role_condition

        base_input_dim = action_dim * 2 + cond_dim + 2
        input_dim = base_input_dim + (role_emb_dim if use_role_condition else 0)
        self.role_emb = nn.Embedding(num_roles, role_emb_dim)
        self.role_head = nn.Sequential(
            nn.Linear(base_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_roles),
        )
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, action_dim)
        self.residual_gate_head = nn.Linear(hidden_dim, action_dim)
        self.arm_update_head = nn.Linear(hidden_dim, 2)

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.residual_gate_head.weight)
        nn.init.constant_(self.residual_gate_head.bias, -4.0)
        nn.init.zeros_(self.arm_update_head.weight)
        nn.init.constant_(self.arm_update_head.bias, 1.0)

    def _arm_features(self, action_chunk):
        left = action_chunk[:, :, self.left_slice]
        right = action_chunk[:, :, self.right_slice]
        left_energy = left.abs().mean(dim=-1, keepdim=True)
        right_energy = right.abs().mean(dim=-1, keepdim=True)
        return torch.cat([left_energy, right_energy], dim=-1)

    def _expand_arm_gate(self, arm_gate):
        batch_size, horizon, _ = arm_gate.shape
        action_gate = torch.zeros(batch_size, horizon, self.action_dim, device=arm_gate.device)
        action_gate[:, :, self.left_slice] = arm_gate[:, :, 0:1]
        action_gate[:, :, self.right_slice] = arm_gate[:, :, 1:2]
        return action_gate

    @staticmethod
    def _soft_hold(action_chunk, action_gate):
        held = [action_chunk[:, 0]]
        for step in range(1, action_chunk.shape[1]):
            update = action_gate[:, step]
            held.append(update * action_chunk[:, step] + (1.0 - update) * held[-1])
        return torch.stack(held, dim=1)

    def forward(self, action_chunk, cond=None, role_id=None):
        batch_size, horizon, action_dim = action_chunk.shape
        if horizon != self.horizon:
            raise ValueError(f"Expected horizon {self.horizon}, got {horizon}")
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim {self.action_dim}, got {action_dim}")
        if self.cond_dim > 0 and cond is None:
            raise ValueError("cond must be provided when cond_dim > 0")

        diff = torch.zeros_like(action_chunk)
        diff[:, 1:] = action_chunk[:, 1:] - action_chunk[:, :-1]
        features = [action_chunk, diff, self._arm_features(diff)]
        if self.cond_dim > 0:
            features.append(cond[:, None, :].expand(batch_size, horizon, -1))

        base_features = torch.cat(features, dim=-1)
        pooled_features = base_features.mean(dim=1)
        role_logits = self.role_head(pooled_features)
        if role_id is None:
            role_prob = torch.softmax(role_logits, dim=-1)
            role_feat = role_prob @ self.role_emb.weight
        else:
            role_id = role_id.to(action_chunk.device).long().view(batch_size).clamp(0, self.num_roles - 1)
            role_prob = torch.nn.functional.one_hot(role_id, self.num_roles).float()
            role_feat = self.role_emb(role_id)
        if self.use_role_condition:
            role_feat = role_feat[:, None, :].expand(batch_size, horizon, -1)
            base_features = torch.cat([base_features, role_feat], dim=-1)

        hidden = self.trunk(base_features)
        delta = self.delta_head(hidden) * self.residual_scale
        residual_gate = torch.sigmoid(self.residual_gate_head(hidden))
        update_prob = torch.sigmoid(self.arm_update_head(hidden))
        first_update = torch.ones(batch_size, 1, 2, device=update_prob.device)
        update_prob = torch.cat([first_update, update_prob[:, 1:]], dim=1)

        if self.hard_inference and not self.training:
            update_gate = (update_prob >= self.update_threshold).float()
            update_gate = torch.cat([first_update, update_gate[:, 1:]], dim=1)
        else:
            update_gate = update_prob

        action_update_gate = self._expand_arm_gate(update_gate)
        freq_action = self._soft_hold(action_chunk, action_update_gate)
        final_action = freq_action + action_update_gate * residual_gate * delta

        info = {
            "base": action_chunk,
            "freq": freq_action,
            "delta": delta,
            "gate": residual_gate,
            "update_prob": update_prob,
            "update_gate": update_gate,
            "action_update_gate": action_update_gate,
            "role_logits": role_logits,
            "role_prob": role_prob,
            "frequency": {
                "left_update_gate": update_gate[:, :, 0:1],
                "right_update_gate": update_gate[:, :, 1:2],
                "left_update_prob": update_prob[:, :, 0:1],
                "right_update_prob": update_prob[:, :, 1:2],
                "left_slice": _slice_to_tuple(self.left_slice),
                "right_slice": _slice_to_tuple(self.right_slice),
            },
        }
        return final_action, info


def bernoulli_kl(q, p, eps=1e-6):
    q = q.clamp(eps, 1.0 - eps)
    p = p.clamp(eps, 1.0 - eps)
    return q * (q.log() - p.log()) + (1.0 - q) * ((1.0 - q).log() - (1.0 - p).log())


def motion_energy_update_target(actions, left_slice, right_slice, temperature=8.0, eps=1e-6):
    """
    Builds soft update targets from synchronous demonstrations.

    Large temporal action changes imply high update probability; near-static
    segments imply low update probability. The first step is always an update.
    """
    left = actions[:, :, left_slice]
    right = actions[:, :, right_slice]

    left_energy = torch.zeros(actions.shape[0], actions.shape[1], 1, device=actions.device)
    right_energy = torch.zeros_like(left_energy)
    left_energy[:, 1:] = (left[:, 1:] - left[:, :-1]).abs().mean(dim=-1, keepdim=True)
    right_energy[:, 1:] = (right[:, 1:] - right[:, :-1]).abs().mean(dim=-1, keepdim=True)

    energy = torch.cat([left_energy, right_energy], dim=-1)
    baseline = energy[:, 1:].mean(dim=1, keepdim=True).detach().clamp(min=eps)
    normalized = energy / baseline
    target = torch.sigmoid(temperature * (normalized - 1.0))
    target[:, 0] = 1.0
    return target.detach()


def motion_role_target(actions, left_slice, right_slice, ratio=1.25, static_eps=1e-4):
    """
    Builds coarse role labels from synchronous demonstrations.

    0: balanced, 1: left primary, 2: right primary, 3: both mostly static.
    These are pseudo-labels, so downstream code should treat them as weak
    supervision rather than ground truth task semantics.
    """
    left = actions[:, :, left_slice]
    right = actions[:, :, right_slice]

    left_energy = (left[:, 1:] - left[:, :-1]).abs().mean(dim=(1, 2))
    right_energy = (right[:, 1:] - right[:, :-1]).abs().mean(dim=(1, 2))

    target = torch.zeros(actions.shape[0], dtype=torch.long, device=actions.device)
    both_static = (left_energy + right_energy) < static_eps
    left_primary = left_energy > right_energy * ratio
    right_primary = right_energy > left_energy * ratio

    target[left_primary] = 1
    target[right_primary] = 2
    target[both_static] = 3
    return target.detach()
