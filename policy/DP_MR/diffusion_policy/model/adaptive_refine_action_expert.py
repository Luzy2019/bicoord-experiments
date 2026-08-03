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
        self.hidden = None

    def reset(self):
        self.plan = None
        self.hidden = None

    def get(
        self,
        plan_shape: Tuple[int, ...],
        hidden_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ):
        plan = None
        hidden = None
        if self.plan is not None and tuple(self.plan.shape) == tuple(plan_shape):
            plan = self.plan.to(device=device, dtype=dtype)
        if self.hidden is not None and tuple(self.hidden.shape) == tuple(hidden_shape):
            hidden = self.hidden.to(device=device, dtype=dtype)
        return plan, hidden

    def update(self, plan: Optional[torch.Tensor] = None, hidden: Optional[torch.Tensor] = None):
        if plan is not None:
            self.plan = plan.detach().cpu()
        if hidden is not None:
            self.hidden = hidden.detach().cpu()


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
            "moe_probs": probs,
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

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        cond_dim: int,
        left_action_dim: Optional[int] = None,
        right_action_dim: Optional[int] = None,
        left_action_start: int = 0,
        right_action_start: Optional[int] = None,
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
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.cond_dim = int(cond_dim)
        if left_action_dim is None:
            left_action_dim = self.action_dim // 2
        if right_action_dim is None:
            right_action_dim = self.action_dim - int(left_action_dim)
        if right_action_start is None:
            right_action_start = int(left_action_start) + int(left_action_dim)
        self.arm_specs = {
            "left": {
                "dim": int(left_action_dim),
                "slice": _slice_from_start_dim(int(left_action_start), int(left_action_dim)),
            },
            "right": {
                "dim": int(right_action_dim),
                "slice": _slice_from_start_dim(int(right_action_start), int(right_action_dim)),
            },
        }
        covered = sum(spec["dim"] for spec in self.arm_specs.values())
        if covered != self.action_dim:
            raise ValueError(f"left/right action dims must sum to action_dim={self.action_dim}, got {covered}")
        self.coarse_plan_steps = max(1, int(coarse_plan_steps))
        self.coarse_plan_steps = min(self.coarse_plan_steps, self.horizon)
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
        self.resolution_levels = self._build_resolution_levels()

        self.action_emb = nn.Linear(self.action_dim, self.hidden_dim)
        self.coarse_emb = nn.Linear(self.action_dim, self.hidden_dim)
        self.cache_plan_emb = nn.Linear(self.action_dim, self.hidden_dim)
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
        self.pos_emb = nn.Parameter(torch.zeros(1, self.horizon, self.hidden_dim))
        self.coarse_pos_emb = nn.Parameter(torch.zeros(1, self.coarse_plan_steps, self.hidden_dim))
        self.round_emb = nn.Embedding(self.max_refine_rounds, self.hidden_dim)
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
        self.arm_heads = nn.ModuleDict({
            arm: nn.Linear(self.hidden_dim, spec["dim"])
            for arm, spec in self.arm_specs.items()
        })
        self.cache = TrajectoryCache()
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.coarse_pos_emb, mean=0.0, std=0.02)

    def _build_resolution_levels(self) -> List[int]:
        if self.max_refine_rounds <= 1:
            return [self.horizon]
        levels = []
        for idx in range(self.max_refine_rounds):
            alpha = float(idx) / float(self.max_refine_rounds - 1)
            steps = round((1.0 - alpha) * self.coarse_plan_steps + alpha * self.horizon)
            if levels:
                steps = max(int(steps), levels[-1] + 1)
            levels.append(min(int(steps), self.horizon))
        levels[-1] = self.horizon
        return levels

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

    def _sparse_attention_mask(self, device, dtype) -> Optional[torch.Tensor]:
        if not self.enable_sparse_attention:
            return None
        idx = torch.arange(self.horizon, device=device)
        query = idx[:, None]
        key = idx[None, :]
        local = (query - key).abs() <= max(self.sparse_attention_band, 0)
        front = key < max(1, self.sparse_attention_band)
        allowed = local | front
        mask = torch.zeros(self.horizon, self.horizon, device=device, dtype=dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask

    def _layer_keep(self, budget_probs: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self.enable_layer_skip:
            return torch.ones(budget_probs.shape[0], device=budget_probs.device, dtype=budget_probs.dtype)
        budget_values = torch.tensor([0.34, 0.67, 1.0], device=budget_probs.device, dtype=budget_probs.dtype)
        expected_budget = (budget_probs * budget_values[None, :]).sum(dim=-1)
        layer_pos = float(layer_idx + 1) / float(max(self.n_layer, 1))
        return torch.sigmoid((expected_budget - layer_pos + 0.10) * 8.0)

    def _predicted_rounds(self, gates: Dict[str, torch.Tensor]) -> torch.Tensor:
        round_choice = gates["round_probs"].argmax(dim=-1) + 1
        need_refine = gates["need_refine_prob"] > 0.5
        min_rounds = torch.full_like(round_choice, self.min_refine_rounds)
        rounds = torch.where(need_refine, round_choice, min_rounds)
        return rounds.clamp(min=self.min_refine_rounds, max=self.max_refine_rounds)

    def _arm_action_mask(self, arm_gates: Dict[str, Dict[str, torch.Tensor]], key: str) -> torch.Tensor:
        batch_size = next(iter(arm_gates.values()))[key].shape[0]
        device = next(iter(arm_gates.values()))[key].device
        dtype = next(iter(arm_gates.values()))[key].dtype
        mask = torch.zeros(batch_size, 1, self.action_dim, device=device, dtype=dtype)
        for arm, spec in self.arm_specs.items():
            value = arm_gates[arm][key]
            if value.ndim == 1:
                value = value[:, None, None].expand(-1, 1, spec["dim"])
            elif value.ndim == 2:
                value = value[:, None, :]
            mask[:, :, spec["slice"]] = value
        return mask

    def _combined_budget_probs(self, arm_gates: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        budget = torch.stack([gate["budget_probs"] for gate in arm_gates.values()], dim=1)
        budget = budget.max(dim=1).values
        return budget / budget.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def _split_head_output(
        self,
        hidden: torch.Tensor,
        arm_gates: Dict[str, Dict[str, torch.Tensor]],
        round_idx: int,
    ) -> torch.Tensor:
        out = torch.zeros(
            hidden.shape[0],
            hidden.shape[1],
            self.action_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        steps = self.resolution_levels[min(int(round_idx), len(self.resolution_levels) - 1)]
        hidden = self._resize_plan(self.out_norm(hidden), steps)
        for arm, spec in self.arm_specs.items():
            arm_out = self.arm_heads[arm](hidden)
            if self.enable_pruning:
                arm_keep = 0.5 + 0.5 * arm_gates[arm]["subspace_keep"][:, None, :]
                arm_out = arm_out * arm_keep
            out[:, :, spec["slice"]] = self._resize_plan(arm_out, self.horizon)
        return out

    def _action_steps_from_rounds(self, rounds: torch.Tensor) -> torch.Tensor:
        levels = torch.tensor(self.resolution_levels, device=rounds.device, dtype=torch.long)
        idx = (rounds.long() - 1).clamp(0, len(self.resolution_levels) - 1)
        return levels[idx]

    def _combine_round_outputs(
        self,
        round_outputs: List[torch.Tensor],
        arm_gates: Dict[str, Dict[str, torch.Tensor]],
        predicted_rounds: Dict[str, torch.Tensor],
        force_full_compute: bool,
    ) -> torch.Tensor:
        stacked = torch.stack(round_outputs, dim=1)
        final = torch.zeros_like(round_outputs[-1])
        for arm, spec in self.arm_specs.items():
            arm_stack = stacked[:, :, :, spec["slice"]]
            if self.training or force_full_compute:
                round_probs = arm_gates[arm]["round_probs"][:, : arm_stack.shape[1]]
                round_probs = round_probs / round_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
                arm_final = (arm_stack * round_probs[:, :, None, None]).sum(dim=1)
            else:
                gather_idx = (predicted_rounds[arm] - 1).clamp_max(arm_stack.shape[1] - 1)
                arm_final = arm_stack[torch.arange(arm_stack.shape[0], device=arm_stack.device), gather_idx]
            final[:, :, spec["slice"]] = arm_final
        return final

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        global_cond: torch.Tensor,
        use_cache: bool = False,
        force_full_compute: bool = False,
        return_rounds: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, horizon, action_dim = sample.shape
        if horizon != self.horizon or action_dim != self.action_dim:
            raise ValueError(
                f"Expected sample shape [B,{self.horizon},{self.action_dim}], got {tuple(sample.shape)}"
            )

        timestep = self._prepare_timestep(timestep, batch_size, sample.device, sample.dtype)
        time_token = self.time_mlp(timestep)
        cond_token = self.cond_mlp(global_cond.to(dtype=sample.dtype))

        coarse_plan = self._resize_plan(sample, self.coarse_plan_steps)
        coarse_tokens = self.coarse_emb(coarse_plan) + self.coarse_pos_emb[:, : self.coarse_plan_steps]
        coarse_summary = coarse_tokens.mean(dim=1)
        summary = self.intent_mlp(time_token + cond_token + coarse_summary)
        arm_gates = {arm: gate(summary) for arm, gate in self.arm_gates.items()}
        combined_budget_probs = self._combined_budget_probs(arm_gates)

        coarse_dense = self._resize_plan(coarse_tokens, self.horizon)
        x = self.action_emb(sample) + self.pos_emb[:, : self.horizon] + coarse_dense
        x = x + time_token[:, None, :] + cond_token[:, None, :] + summary[:, None, :]

        cached_plan = None
        cached_hidden = None
        cache_reuse_mask = torch.zeros(batch_size, 1, self.action_dim, device=sample.device, dtype=sample.dtype)
        cache_reuse = torch.zeros(batch_size, device=sample.device, dtype=sample.dtype)
        if self.enable_cache and use_cache:
            cached_plan, cached_hidden = self.cache.get(
                plan_shape=sample.shape,
                hidden_shape=x.shape,
                device=sample.device,
                dtype=sample.dtype,
            )
            cache_reuse_mask = self._arm_action_mask(arm_gates, "cache_prob")
            cache_reuse = cache_reuse_mask.mean(dim=(1, 2))
            if cached_plan is not None:
                x = x + self.cache_plan_emb(cached_plan * cache_reuse_mask)
            if cached_hidden is not None:
                x = (1.0 - cache_reuse[:, None, None]) * x + cache_reuse[:, None, None] * cached_hidden

        if self.training or force_full_compute:
            rounds_to_run = self.max_refine_rounds
            predicted_rounds = {arm: self._predicted_rounds(gate) for arm, gate in arm_gates.items()}
        else:
            predicted_rounds = {arm: self._predicted_rounds(gate) for arm, gate in arm_gates.items()}
            rounds_to_run = int(torch.stack(list(predicted_rounds.values()), dim=1).max().item())

        attn_mask = self._sparse_attention_mask(sample.device, sample.dtype)
        round_outputs: List[torch.Tensor] = []
        exit_events = []
        layer_keep_values = []
        token_keep_values = []
        moe_balance_terms = []
        moe_entropy_terms = []
        actual_layers = 0

        for round_idx in range(rounds_to_run):
            round_id = torch.full((batch_size,), round_idx, device=sample.device, dtype=torch.long)
            x_round = x + self.round_emb(round_id)[:, None, :]
            for layer_idx, block in enumerate(self.blocks):
                layer_keep = self._layer_keep(combined_budget_probs, layer_idx)
                if (
                    not self.training
                    and self.enable_layer_skip
                    and layer_keep.mean().item() < 0.05
                ):
                    layer_keep_values.append(layer_keep)
                    continue
                x_round, block_aux = block(
                    x_round,
                    summary,
                    attn_mask=attn_mask,
                    layer_keep=layer_keep,
                    enable_pruning=self.enable_pruning,
                )
                actual_layers += 1
                layer_keep_values.append(layer_keep)
                token_keep_values.append(block_aux["token_keep"].mean())
                moe_balance_terms.append(block_aux["moe_balance_loss"])
                moe_entropy_terms.append(block_aux["moe_entropy"])
                exit_prob = block_aux["exit_prob"]
                exit_events.append(exit_prob)
                if (
                    not self.training
                    and self.enable_early_exit
                    and layer_idx > 0
                    and exit_prob.mean().item() > self.early_exit_threshold
                ):
                    break
            x = x_round
            out = self._split_head_output(x, arm_gates, round_idx)
            round_outputs.append(out)

        final = self._combine_round_outputs(round_outputs, arm_gates, predicted_rounds, force_full_compute)

        if exit_events:
            exit_stack = torch.stack(exit_events, dim=0)
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
            arm_rounds = predicted_rounds[arm].to(dtype=sample.dtype)
            arm_action_steps = self._action_steps_from_rounds(predicted_rounds[arm]).to(dtype=sample.dtype)
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
                f"{arm}_predicted_rounds": arm_rounds,
                f"{arm}_action_steps": arm_action_steps,
                f"{arm}_refine_rounds_used": arm_rounds.mean(),
                f"{arm}_action_steps_used": arm_action_steps.mean(),
                f"{arm}_cache_reuse_rate": gate["cache_prob"].mean(),
                f"{arm}_compute_budget_pred": arm_budget.mean(),
            })
            arm_round_tensor.append(arm_rounds)
            arm_budget_values.append(arm_budget)
            arm_need_values.append(gate["need_refine_prob"])
        predicted_round_tensor = torch.stack(arm_round_tensor, dim=1)
        compute_budget_pred = torch.stack(arm_budget_values, dim=1).mean(dim=1)
        need_refine_prob = torch.stack(arm_need_values, dim=1).mean(dim=1)
        aux = {
            **arm_aux,
            "coarse_plan": coarse_plan,
            "round_outputs": round_outputs if return_rounds else [],
            "predicted_rounds": predicted_round_tensor,
            "need_refine_prob": need_refine_prob,
            "refine_rounds_used": torch.full(
                (batch_size,),
                float(rounds_to_run),
                device=sample.device,
                dtype=sample.dtype,
            ),
            "actual_layers": torch.tensor(float(actual_layers), device=sample.device, dtype=sample.dtype),
            "early_exit_rate": early_exit_rate,
            "exit_prob_mean": exit_prob_mean,
            "cache_reuse_rate": cache_reuse.mean(),
            "layer_keep_mean": layer_keep_mean,
            "token_keep_mean": token_keep_mean,
            "moe_balance_loss": moe_balance_loss,
            "moe_entropy": moe_entropy,
            "compute_budget_pred": compute_budget_pred.mean(),
            "final_hidden": x,
            "cached_plan_available": torch.tensor(
                float(cached_plan is not None),
                device=sample.device,
                dtype=sample.dtype,
            ),
            "cached_hidden_available": torch.tensor(
                float(cached_hidden is not None),
                device=sample.device,
                dtype=sample.dtype,
            ),
        }
        return final, aux
