from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _slice_from_start_dim(start: int, dim: int) -> slice:
    return slice(int(start), int(start) + int(dim))


class SinusoidalStepEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        steps = steps.float()
        half_dim = self.dim // 2
        if half_dim == 0:
            return steps[:, None]
        freq = torch.exp(
            torch.arange(half_dim, device=steps.device, dtype=steps.dtype)
            * -(math.log(10000.0) / max(half_dim - 1, 1))
        )
        emb = steps[:, None] * freq[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class TemporalLossWeights(nn.Module):

    def __init__(self, horizon: int, alpha: float = 0.25):
        super().__init__()
        self.horizon = int(horizon)
        self.alpha = float(alpha)

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        device = device if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        idx = torch.arange(self.horizon, device=device, dtype=dtype)
        weights = torch.exp(-self.alpha * idx)
        weights = weights / weights.mean().clamp_min(1.0e-6)
        return weights.view(1, self.horizon, 1)


class TrajectoryCache:

    def __init__(self):
        self.plan = None

    def reset(self):
        self.plan = None

    def get_plan(
        self,
        plan_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ):
        if self.plan is not None and tuple(self.plan.shape) == tuple(plan_shape):
            return self.plan.to(device=device, dtype=dtype)
        return None

    def update(self, plan: Optional[torch.Tensor] = None, hidden: Optional[torch.Tensor] = None):
        del hidden
        if plan is not None:
            self.plan = plan.detach().cpu()


class RefineGate(nn.Module):

    def __init__(self, hidden_dim: int, max_refine_rounds: int, action_dim: int):
        super().__init__()
        self.max_refine_rounds = int(max_refine_rounds)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.refine_head = nn.Linear(hidden_dim, 1)
        self.round_head = nn.Linear(hidden_dim, self.max_refine_rounds)
        self.budget_head = nn.Linear(hidden_dim, 3)
        self.cache_head = nn.Linear(hidden_dim, 1)
        self.subspace_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, summary: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(summary)
        need_refine_logit = self.refine_head(h).squeeze(-1)
        round_logits = self.round_head(h)
        budget_logits = self.budget_head(h)
        cache_logit = self.cache_head(h).squeeze(-1)
        subspace_logits = self.subspace_head(h)
        return {
            "need_refine_logit": need_refine_logit,
            "need_refine_prob": torch.sigmoid(need_refine_logit),
            "round_logits": round_logits,
            "round_probs": torch.softmax(round_logits, dim=-1),
            "budget_logits": budget_logits,
            "budget_probs": torch.softmax(budget_logits, dim=-1),
            "cache_logit": cache_logit,
            "cache_prob": torch.sigmoid(cache_logit),
            "subspace_logits": subspace_logits,
            "subspace_keep": torch.sigmoid(subspace_logits),
        }


class MoEFeedForward(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        ff_mult: int = 4,
        num_experts: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        inner_dim = int(hidden_dim * ff_mult)
        self.router = nn.Linear(hidden_dim, self.num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, hidden_dim),
            )
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        router_logits = self.router(x)
        probs = torch.softmax(router_logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        y = (expert_outputs * probs.unsqueeze(-1)).sum(dim=-2)
        avg_probs = probs.mean(dim=(0, 1))
        target = torch.full_like(avg_probs, 1.0 / self.num_experts)
        balance_loss = (avg_probs - target).pow(2).mean()
        entropy = -(avg_probs * (avg_probs + 1.0e-8).log()).sum()
        return y, {
            "moe_balance_loss": balance_loss,
            "moe_entropy": entropy,
        }


class DenseFeedForward(nn.Module):

    def __init__(self, hidden_dim: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(hidden_dim * ff_mult)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = x.sum() * 0.0
        return self.net(x), {
            "moe_balance_loss": zero,
            "moe_entropy": zero,
        }


class ConditionalTransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_head: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        enable_moe: bool = True,
        num_moe_experts: int = 4,
    ):
        super().__init__()
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        if enable_moe:
            self.ff = MoEFeedForward(hidden_dim, ff_mult=ff_mult, num_experts=num_moe_experts, dropout=dropout)
        else:
            self.ff = DenseFeedForward(hidden_dim, ff_mult=ff_mult, dropout=dropout)
        self.exit_head = nn.Linear(hidden_dim, 1)
        self.prune_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        cond_summary: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        layer_keep: Optional[torch.Tensor] = None,
        enable_pruning: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = x
        cond = self.cond_proj(cond_summary).unsqueeze(1)
        h = self.norm1(x + cond)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x_candidate = residual + attn_out
        h = self.norm2(x_candidate + cond)
        ff_out, aux = self.ff(h)
        x_candidate = x_candidate + ff_out

        if enable_pruning:
            token_keep = torch.sigmoid(self.prune_head(x_candidate))
            x_candidate = token_keep * x_candidate + (1.0 - token_keep) * residual
        else:
            token_keep = torch.ones_like(x_candidate[..., :1])

        if layer_keep is not None:
            keep = layer_keep.view(-1, 1, 1).to(dtype=x.dtype)
            x_next = keep * x_candidate + (1.0 - keep) * residual
        else:
            x_next = x_candidate

        exit_prob = torch.sigmoid(self.exit_head(x_next).mean(dim=1).squeeze(-1))
        aux.update({
            "exit_prob": exit_prob,
            "token_keep": token_keep,
        })
        return x_next, aux


class AdaptiveRefineActionExpert(nn.Module):
    """Per-arm 10-token action expert with residual multi-round densification."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        cond_dim: int,
        left_action_dim: Optional[int] = None,
        right_action_dim: Optional[int] = None,
        left_action_start: int = 0,
        right_action_start: Optional[int] = None,
        expert_horizon: Optional[int] = None,
        coarse_plan_steps: int = 10,
        max_refine_rounds: int = 3,
        min_refine_rounds: int = 1,
        hidden_dim: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        num_moe_experts: int = 4,
        enable_cache: bool = True,
        enable_early_exit: bool = True,
        enable_layer_skip: bool = True,
        enable_moe: bool = True,
        enable_pruning: bool = True,
        enable_sparse_attention: bool = True,
        early_exit_threshold: float = 0.92,
        sparse_attention_band: int = 4,
        front_refine_alpha: float = 0.25,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.semantic_horizon = int(horizon)
        self.cond_dim = int(cond_dim)
        if left_action_dim is None:
            left_action_dim = self.action_dim // 2
        if right_action_dim is None:
            right_action_dim = self.action_dim - int(left_action_dim)
        if right_action_start is None:
            right_action_start = int(left_action_start) + int(left_action_dim)
        self.arm_specs = {
            "left": {
                "id": 0,
                "dim": int(left_action_dim),
                "slice": _slice_from_start_dim(int(left_action_start), int(left_action_dim)),
            },
            "right": {
                "id": 1,
                "dim": int(right_action_dim),
                "slice": _slice_from_start_dim(int(right_action_start), int(right_action_dim)),
            },
        }
        covered = sum(spec["dim"] for spec in self.arm_specs.values())
        if covered != self.action_dim:
            raise ValueError(f"left/right action dims must sum to action_dim={self.action_dim}, got {covered}")

        if expert_horizon is None:
            expert_horizon = coarse_plan_steps
        self.expert_horizon = max(1, int(expert_horizon))
        self.expert_horizon = min(self.expert_horizon, self.semantic_horizon)
        self.coarse_plan_steps = self.expert_horizon
        self.max_refine_rounds = max(1, int(max_refine_rounds))
        self.min_refine_rounds = max(1, int(min_refine_rounds))
        self.min_refine_rounds = min(self.min_refine_rounds, self.max_refine_rounds)
        self.hidden_dim = int(hidden_dim)
        self.n_layer = int(n_layer)
        self.enable_cache = bool(enable_cache)
        self.enable_early_exit = bool(enable_early_exit)
        self.enable_layer_skip = bool(enable_layer_skip)
        self.enable_moe = bool(enable_moe)
        self.enable_pruning = bool(enable_pruning)
        self.enable_sparse_attention = bool(enable_sparse_attention)
        self.early_exit_threshold = float(early_exit_threshold)
        self.sparse_attention_band = int(sparse_attention_band)
        self.front_refine_alpha = float(front_refine_alpha)

        self.time_mlp = nn.Sequential(
            SinusoidalStepEmbedding(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.intent_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.joint_gate_emb = nn.Linear(self.action_dim, self.hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.expert_horizon, self.hidden_dim))
        self.round_emb = nn.Embedding(self.max_refine_rounds, self.hidden_dim)
        self.arm_emb = nn.Embedding(len(self.arm_specs), self.hidden_dim)
        self.arm_action_emb = nn.ModuleDict({
            arm: nn.Linear(spec["dim"], self.hidden_dim)
            for arm, spec in self.arm_specs.items()
        })
        self.arm_delta_heads = nn.ModuleDict({
            arm: nn.Linear(self.hidden_dim, spec["dim"])
            for arm, spec in self.arm_specs.items()
        })
        self.arm_gates = nn.ModuleDict({
            arm: RefineGate(self.hidden_dim, self.max_refine_rounds, spec["dim"])
            for arm, spec in self.arm_specs.items()
        })
        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(
                hidden_dim=self.hidden_dim,
                n_head=n_head,
                ff_mult=ff_mult,
                dropout=dropout,
                enable_moe=self.enable_moe,
                num_moe_experts=num_moe_experts,
            )
            for _ in range(self.n_layer)
        ])
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.cache = TrajectoryCache()
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def reset_cache(self):
        self.cache.reset()

    def update_cache(self, plan: Optional[torch.Tensor], hidden: Optional[torch.Tensor]):
        self.cache.update(plan=plan, hidden=hidden)

    def _prepare_timestep(self, timestep, batch_size: int, device, dtype):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(device=device)
        else:
            timestep = timestep.to(device=device)
        timestep = timestep.expand(batch_size).to(dtype=dtype)
        return timestep

    def _resize_plan(self, plan: torch.Tensor, steps: int) -> torch.Tensor:
        if plan.shape[1] == steps:
            return plan
        if steps == 1:
            return plan.mean(dim=1, keepdim=True)
        return F.interpolate(
            plan.transpose(1, 2),
            size=steps,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

    def _front_refine_weights(self, batch_size: int, dim: int, device, dtype) -> torch.Tensor:
        idx = torch.arange(self.expert_horizon, device=device, dtype=dtype)
        weights = torch.exp(-self.front_refine_alpha * idx)
        weights = weights / weights.max().clamp_min(1.0e-6)
        return weights.view(1, self.expert_horizon, 1).expand(batch_size, -1, dim)

    def _sparse_attention_mask(self, device, dtype) -> Optional[torch.Tensor]:
        if not self.enable_sparse_attention:
            return None
        idx = torch.arange(self.expert_horizon, device=device)
        query = idx[:, None]
        key = idx[None, :]
        local = (query - key).abs() <= max(self.sparse_attention_band, 0)
        front = key < max(1, self.sparse_attention_band)
        allowed = local | front
        mask = torch.zeros(self.expert_horizon, self.expert_horizon, device=device, dtype=dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask

    def _layer_keep(self, budget_probs: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self.enable_layer_skip:
            return torch.ones(budget_probs.shape[0], device=budget_probs.device, dtype=budget_probs.dtype)
        budget_values = torch.tensor([0.34, 0.67, 1.0], device=budget_probs.device, dtype=budget_probs.dtype)
        expected_budget = (budget_probs * budget_values[None, :]).sum(dim=-1)
        layer_pos = float(layer_idx + 1) / float(max(self.n_layer, 1))
        return torch.sigmoid((expected_budget - layer_pos + 0.10) * 8.0)

    def _combined_budget_probs(self, arm_gates: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        budget = torch.stack([gate["budget_probs"] for gate in arm_gates.values()], dim=1)
        budget = budget.max(dim=1).values
        return budget / budget.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def _predicted_rounds(self, gates: Dict[str, torch.Tensor]) -> torch.Tensor:
        round_choice = gates["round_probs"].argmax(dim=-1) + 1
        need_refine = gates["need_refine_prob"] > 0.5
        min_rounds = torch.full_like(round_choice, self.min_refine_rounds)
        rounds = torch.where(need_refine, round_choice, min_rounds)
        return rounds.clamp(min=self.min_refine_rounds, max=self.max_refine_rounds)

    def _coerce_forced_rounds(self, value, batch_size: int, device) -> Optional[Dict[str, torch.Tensor]]:
        if value is None:
            return None
        result = {}
        if torch.is_tensor(value):
            if value.ndim == 1:
                value = value[:, None].expand(-1, len(self.arm_specs))
            for idx, arm in enumerate(self.arm_specs.keys()):
                result[arm] = value[:, idx].to(device=device, dtype=torch.long)
        elif isinstance(value, dict):
            for arm in self.arm_specs.keys():
                arm_value = value.get(arm, self.min_refine_rounds)
                if torch.is_tensor(arm_value):
                    result[arm] = arm_value.to(device=device, dtype=torch.long).expand(batch_size)
                else:
                    result[arm] = torch.full((batch_size,), int(arm_value), device=device, dtype=torch.long)
        else:
            scalar = int(value)
            for arm in self.arm_specs.keys():
                result[arm] = torch.full((batch_size,), scalar, device=device, dtype=torch.long)
        for arm in result:
            result[arm] = result[arm].clamp(min=self.min_refine_rounds, max=self.max_refine_rounds)
        return result

    def compute_gate_state(
        self,
        sample: torch.Tensor,
        timestep,
        global_cond: torch.Tensor,
        use_cache: bool = False,
        forced_rounds=None,
    ) -> Dict[str, object]:
        batch_size, horizon, action_dim = sample.shape
        if horizon != self.semantic_horizon or action_dim != self.action_dim:
            raise ValueError(
                f"Expected gate sample shape [B,{self.semantic_horizon},{self.action_dim}], got {tuple(sample.shape)}"
            )

        gate_sample = sample
        cached_plan = None
        if self.enable_cache and use_cache:
            cached_plan = self.cache.get_plan(
                plan_shape=sample.shape,
                device=sample.device,
                dtype=sample.dtype,
            )
            if cached_plan is not None:
                gate_sample = cached_plan

        timestep = self._prepare_timestep(timestep, batch_size, sample.device, sample.dtype)
        time_token = self.time_mlp(timestep)
        cond_token = self.cond_mlp(global_cond.to(dtype=sample.dtype))
        coarse_joint = self._resize_plan(gate_sample, self.expert_horizon)
        coarse_summary = self.joint_gate_emb(coarse_joint).mean(dim=1)
        summary = self.intent_mlp(time_token + cond_token + coarse_summary)
        arm_gates = {arm: gate(summary) for arm, gate in self.arm_gates.items()}
        predicted_rounds = {arm: self._predicted_rounds(gate) for arm, gate in arm_gates.items()}
        forced = self._coerce_forced_rounds(forced_rounds, batch_size, sample.device)
        used_rounds = forced if forced is not None else predicted_rounds
        return {
            "summary": summary,
            "arm_gates": arm_gates,
            "predicted_rounds": predicted_rounds,
            "used_rounds": used_rounds,
            "combined_budget_probs": self._combined_budget_probs(arm_gates),
            "cached_plan_available": torch.tensor(
                float(cached_plan is not None),
                device=sample.device,
                dtype=sample.dtype,
            ),
        }

    def _run_arm_expert(
        self,
        arm: str,
        arm_input: torch.Tensor,
        timestep: torch.Tensor,
        summary: torch.Tensor,
        budget_probs: torch.Tensor,
        round_idx: int,
        attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]], torch.Tensor, int]:
        spec = self.arm_specs[arm]
        batch_size = arm_input.shape[0]
        arm_id = torch.full((batch_size,), spec["id"], device=arm_input.device, dtype=torch.long)
        round_id = torch.full((batch_size,), round_idx, device=arm_input.device, dtype=torch.long)
        x = self.arm_action_emb[arm](arm_input)
        x = x + self.pos_emb[:, : self.expert_horizon]
        x = x + self.round_emb(round_id)[:, None, :]
        x = x + self.arm_emb(arm_id)[:, None, :]
        x = x + timestep[:, None, :] + summary[:, None, :]

        aux = {
            "exit_events": [],
            "layer_keep_values": [],
            "token_keep_values": [],
            "moe_balance_terms": [],
            "moe_entropy_terms": [],
        }
        actual_layers = 0
        for layer_idx, block in enumerate(self.blocks):
            layer_keep = self._layer_keep(budget_probs, layer_idx)
            if (
                not self.training
                and self.enable_layer_skip
                and layer_keep.mean().item() < 0.05
            ):
                aux["layer_keep_values"].append(layer_keep)
                continue
            x, block_aux = block(
                x,
                summary,
                attn_mask=attn_mask,
                layer_keep=layer_keep,
                enable_pruning=self.enable_pruning,
            )
            actual_layers += 1
            aux["layer_keep_values"].append(layer_keep)
            aux["token_keep_values"].append(block_aux["token_keep"].mean())
            aux["moe_balance_terms"].append(block_aux["moe_balance_loss"])
            aux["moe_entropy_terms"].append(block_aux["moe_entropy"])
            exit_prob = block_aux["exit_prob"]
            aux["exit_events"].append(exit_prob)
            if (
                not self.training
                and self.enable_early_exit
                and layer_idx > 0
                and exit_prob.mean().item() > self.early_exit_threshold
            ):
                break
        delta = self.arm_delta_heads[arm](self.out_norm(x))
        return delta, aux, x, actual_layers

    def _densify_round_lows(self, lows: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(lows, dim=1)
        interleaved = stacked.permute(0, 2, 1, 3).reshape(
            stacked.shape[0],
            stacked.shape[1] * stacked.shape[2],
            stacked.shape[3],
        )
        return self._resize_plan(interleaved, self.semantic_horizon)

    def _action_steps_from_rounds(self, rounds: torch.Tensor) -> torch.Tensor:
        return (rounds.long() * self.expert_horizon).clamp(max=self.semantic_horizon)

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        global_cond: torch.Tensor,
        use_cache: bool = False,
        force_full_compute: bool = False,
        return_rounds: bool = True,
        gate_state: Optional[Dict[str, object]] = None,
        forced_rounds=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, horizon, action_dim = sample.shape
        if horizon != self.semantic_horizon or action_dim != self.action_dim:
            raise ValueError(
                f"Expected sample shape [B,{self.semantic_horizon},{self.action_dim}], got {tuple(sample.shape)}"
            )

        if gate_state is None:
            gate_state = self.compute_gate_state(
                sample=sample,
                timestep=timestep,
                global_cond=global_cond,
                use_cache=use_cache,
                forced_rounds=forced_rounds,
            )
            gate_reuse_rate = sample.sum() * 0.0
        else:
            gate_reuse_rate = torch.ones((), device=sample.device, dtype=sample.dtype)

        timestep = self._prepare_timestep(timestep, batch_size, sample.device, sample.dtype)
        time_token = self.time_mlp(timestep)
        summary = gate_state["summary"].to(dtype=sample.dtype)
        arm_gates = gate_state["arm_gates"]
        combined_budget_probs = gate_state["combined_budget_probs"].to(dtype=sample.dtype)
        if force_full_compute:
            used_rounds = {
                arm: torch.full(
                    (batch_size,),
                    self.max_refine_rounds,
                    device=sample.device,
                    dtype=torch.long,
                )
                for arm in self.arm_specs.keys()
            }
        elif forced_rounds is not None:
            used_rounds = self._coerce_forced_rounds(forced_rounds, batch_size, sample.device)
        else:
            used_rounds = gate_state["used_rounds"]

        attn_mask = self._sparse_attention_mask(sample.device, sample.dtype)
        round_outputs: List[torch.Tensor] = []
        exit_events = []
        layer_keep_values = []
        token_keep_values = []
        moe_balance_terms = []
        moe_entropy_terms = []
        actual_layers = 0
        final_hidden = None
        active_token_count = sample.sum() * 0.0

        arm_lows: Dict[str, List[torch.Tensor]] = {}
        arm_current_dense: Dict[str, torch.Tensor] = {}
        arm_initial_low: Dict[str, torch.Tensor] = {}
        for arm, spec in self.arm_specs.items():
            arm_slice = spec["slice"]
            arm_initial_low[arm] = self._resize_plan(sample[:, :, arm_slice], self.expert_horizon)
            arm_lows[arm] = [
                torch.zeros(
                    batch_size,
                    self.expert_horizon,
                    spec["dim"],
                    device=sample.device,
                    dtype=sample.dtype,
                )
                for _ in range(self.max_refine_rounds)
            ]
            arm_current_dense[arm] = torch.zeros(
                batch_size,
                self.semantic_horizon,
                spec["dim"],
                device=sample.device,
                dtype=sample.dtype,
            )

        rounds_to_run = int(torch.stack(list(used_rounds.values()), dim=1).max().item())
        for round_idx in range(rounds_to_run):
            dense_round = torch.zeros(
                batch_size,
                self.semantic_horizon,
                self.action_dim,
                device=sample.device,
                dtype=sample.dtype,
            )
            for arm, spec in self.arm_specs.items():
                arm_rounds = used_rounds[arm].long()
                active_mask = arm_rounds > round_idx
                if active_mask.any():
                    active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
                    if round_idx == 0:
                        arm_input = arm_initial_low[arm][active_idx]
                    else:
                        arm_input = self._resize_plan(arm_current_dense[arm][active_idx], self.expert_horizon)
                    delta, block_aux, hidden, layers = self._run_arm_expert(
                        arm=arm,
                        arm_input=arm_input,
                        timestep=time_token[active_idx],
                        summary=summary[active_idx],
                        budget_probs=combined_budget_probs[active_idx],
                        round_idx=round_idx,
                        attn_mask=attn_mask,
                    )
                    gate = arm_gates[arm]
                    if self.enable_pruning:
                        keep = 0.5 + 0.5 * gate["subspace_keep"][active_idx, None, :]
                        delta = delta * keep
                    if round_idx == 0:
                        low = delta
                    else:
                        front_weight = self._front_refine_weights(
                            delta.shape[0],
                            spec["dim"],
                            delta.device,
                            delta.dtype,
                        )
                        low = arm_input + front_weight * delta
                    arm_lows[arm][round_idx][active_idx] = low
                    lows = [arm_lows[arm][idx][active_idx] for idx in range(round_idx + 1)]
                    arm_current_dense[arm][active_idx] = self._densify_round_lows(lows)
                    active_token_count = active_token_count + float(self.expert_horizon) * active_idx.numel()
                    actual_layers += layers
                    final_hidden = hidden
                    for key, target in (
                        ("exit_events", exit_events),
                        ("layer_keep_values", layer_keep_values),
                        ("token_keep_values", token_keep_values),
                        ("moe_balance_terms", moe_balance_terms),
                        ("moe_entropy_terms", moe_entropy_terms),
                    ):
                        target.extend(block_aux[key])
                dense_round[:, :, spec["slice"]] = arm_current_dense[arm]
            round_outputs.append(dense_round)

        final = round_outputs[-1] if round_outputs else torch.zeros_like(sample)

        if exit_events:
            exit_stack = torch.cat([v.reshape(-1) for v in exit_events], dim=0)
            early_exit_rate = (exit_stack > self.early_exit_threshold).float().mean()
            exit_prob_mean = exit_stack.mean()
        else:
            early_exit_rate = sample.sum() * 0.0
            exit_prob_mean = sample.sum() * 0.0
        if layer_keep_values:
            layer_keep_mean = torch.stack([v.mean() for v in layer_keep_values]).mean()
        else:
            layer_keep_mean = sample.sum() * 0.0
        if token_keep_values:
            token_keep_mean = torch.stack(token_keep_values).mean()
        else:
            token_keep_mean = sample.sum() * 0.0
        if moe_balance_terms:
            moe_balance_loss = torch.stack(moe_balance_terms).mean()
            moe_entropy = torch.stack(moe_entropy_terms).mean()
        else:
            moe_balance_loss = sample.sum() * 0.0
            moe_entropy = sample.sum() * 0.0

        budget_values = torch.tensor([0.34, 0.67, 1.0], device=sample.device, dtype=sample.dtype)
        arm_aux = {}
        arm_round_tensor = []
        arm_budget_values = []
        arm_need_values = []
        for arm, gate in arm_gates.items():
            arm_rounds = used_rounds[arm].to(dtype=sample.dtype)
            predicted_rounds = gate_state["predicted_rounds"][arm].to(dtype=sample.dtype)
            arm_action_steps = self._action_steps_from_rounds(used_rounds[arm]).to(dtype=sample.dtype)
            arm_budget = (gate["budget_probs"] * budget_values[None, :]).sum(dim=-1)
            arm_aux.update({
                f"{arm}_need_refine_logit": gate["need_refine_logit"],
                f"{arm}_need_refine_prob": gate["need_refine_prob"],
                f"{arm}_round_logits": gate["round_logits"],
                f"{arm}_round_probs": gate["round_probs"],
                f"{arm}_budget_logits": gate["budget_logits"],
                f"{arm}_budget_probs": gate["budget_probs"],
                f"{arm}_cache_logit": gate["cache_logit"],
                f"{arm}_cache_prob": gate["cache_prob"],
                f"{arm}_subspace_logits": gate["subspace_logits"],
                f"{arm}_subspace_keep": gate["subspace_keep"],
                f"{arm}_predicted_rounds": predicted_rounds,
                f"{arm}_used_rounds": arm_rounds,
                f"{arm}_action_steps": arm_action_steps,
                f"{arm}_refine_rounds_used": arm_rounds.mean(),
                f"{arm}_action_steps_used": arm_action_steps.mean(),
                f"{arm}_cache_reuse_rate": gate["cache_prob"].mean(),
                f"{arm}_compute_budget_pred": arm_budget.mean(),
            })
            arm_round_tensor.append(arm_rounds)
            arm_budget_values.append(arm_budget)
            arm_need_values.append(gate["need_refine_prob"])

        used_round_tensor = torch.stack(arm_round_tensor, dim=1)
        compute_budget_pred = torch.stack(arm_budget_values, dim=1).mean(dim=1)
        need_refine_prob = torch.stack(arm_need_values, dim=1).mean(dim=1)
        cached_plan_available = gate_state.get(
            "cached_plan_available",
            torch.tensor(0.0, device=sample.device, dtype=sample.dtype),
        )
        cache_reuse_rate = sample.sum() * 0.0
        if cached_plan_available.detach().float().item() > 0.0:
            cache_reuse_rate = torch.stack([gate["cache_prob"].mean() for gate in arm_gates.values()]).mean()
        aux = {
            **arm_aux,
            "coarse_plan": self._resize_plan(sample, self.expert_horizon),
            "round_outputs": round_outputs if return_rounds else [],
            "predicted_rounds": used_round_tensor,
            "need_refine_prob": need_refine_prob,
            "refine_rounds_used": used_round_tensor.max(dim=1).values,
            "actual_layers": torch.tensor(float(actual_layers), device=sample.device, dtype=sample.dtype),
            "early_exit_rate": early_exit_rate,
            "exit_prob_mean": exit_prob_mean,
            "cache_reuse_rate": cache_reuse_rate,
            "gate_reuse_rate": gate_reuse_rate,
            "layer_keep_mean": layer_keep_mean,
            "token_keep_mean": token_keep_mean,
            "moe_balance_loss": moe_balance_loss,
            "moe_entropy": moe_entropy,
            "compute_budget_pred": compute_budget_pred.mean(),
            "active_expert_tokens": active_token_count / float(max(batch_size, 1)),
            "expert_horizon": torch.tensor(float(self.expert_horizon), device=sample.device, dtype=sample.dtype),
            "final_hidden": final_hidden if final_hidden is not None else sample.sum() * 0.0,
            "cached_plan_available": cached_plan_available,
            "cached_hidden_available": torch.tensor(0.0, device=sample.device, dtype=sample.dtype),
        }
        return final, aux
