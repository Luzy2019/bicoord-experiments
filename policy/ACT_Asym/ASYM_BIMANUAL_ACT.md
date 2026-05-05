# Asymmetric Bimanual ACT

This variant keeps the original ACT transformer as the base policy and adds a
frequency-aware residual adapter after the action chunk.

## Model

ACT still predicts a unified high-frequency action chunk:

```text
A_base = ACT(obs, qpos)              # [B, H, D]
D = D_left + D_right
```

The frequency projector keeps the same tensor shape but imposes per-arm update
strides. For example, with `left_stride=1` and `right_stride=5`, the right arm
action is held for four intermediate steps:

```text
A_freq = FrequencyProject(A_base; K_left, K_right)
```

The residual head then predicts a correction and a gate:

```text
delta, gate = ResidualAdapter(A_freq, qpos, phase_left, phase_right, mask_left, mask_right)
A_final = A_freq + residual_mask * gate * delta
```

In strict mode, `residual_mask` is the arm update mask. This means the
low-frequency arm is only changed on its update steps. If strict mode is
disabled, the low-frequency base action is held, but residual can make
high-frequency micro-corrections.

## Output Format

The model output remains compatible with the original ACT code:

```text
A_final: [B, H, D]
```

For a 14-DoF bimanual action:

```text
left  = A_final[:, :, 0:7]
right = A_final[:, :, 7:14]
```

The adapter also stores auxiliary tensors:

```text
base, freq, delta, gate, residual_mask
left_update_mask, right_update_mask
left_phase, right_phase
```

## Training With Existing Synchronous Data

Existing ACT datasets usually contain synchronous left/right actions. The
adapter can reuse these datasets directly by projecting the demonstration chunk
onto the same asymmetric schedule used by the model:

```text
A_target = FrequencyProject(A_gt; K_left, K_right)
L_bc = L1(A_final, A_target)
```

This is the default `asym_target_mode=projected`. It is the recommended setting
for strict asymmetric execution because the target is physically executable by
the chosen update masks.

If you disable strict execution and want a low-frequency base trajectory plus
high-frequency residual correction, use `asym_target_mode=original` so the model
is still supervised by the original high-frequency demonstration.

The total training objective is:

```text
L = L_bc
  + kl_weight * L_kl
  + lambda_res * ||delta||^2
  + lambda_gate * ||gate||_1
  + lambda_smooth * ||A_final[t+1] - A_final[t]||^2
  + lambda_freq * ||right[t] - right[t-1]||^2 on non-update right-arm steps
```

`lambda_freq` is useful for strict low-frequency execution. Set it smaller, or
disable strict execution, when you want low-frequency base actions plus
high-frequency residual correction.

## Recommended First Experiment

Train or fine-tune with:

```bash
python imitate_episodes.py \
  --policy_class ACT \
  --use_asym_residual \
  --left_stride 1 \
  --right_stride 5 \
  --asym_projector_mode hold \
  --asym_target_mode projected \
  --asym_lambda_res 1e-4 \
  --asym_lambda_gate 1e-4 \
  --asym_lambda_smooth 1e-3 \
  --asym_lambda_freq 1e-2 \
  ...
```

This corresponds to unified high-frequency inference and strict asymmetric
execution: the left arm updates every step, while the right arm updates every
five steps.

To reverse the frequencies, set:

```bash
--left_stride 5 --right_stride 1
```

To train one model with multiple possible frequency patterns, use:

```bash
--asym_stride_choices "1,1;1,5;5,1"
```

During training, one pair is sampled for each batch. This covers equal-frequency,
left-high/right-low, and left-low/right-high cases using the same synchronous
dataset.

If the important arm should vary with the trajectory instead of a preset stride,
use learned scheduling:

```bash
--asym_schedule_mode learned --asym_target_mode original
```

This predicts soft left/right update gates from the synchronous action sequence
and turns them into hard update decisions at inference.

For the alternative mode, where the right arm has a low-frequency base action
but can still receive high-frequency residual corrections, add:

```bash
--asym_highfreq_residual
```

## Checkpoint Compatibility

When `use_asym_residual=true`, old ACT checkpoints can be loaded with
`strict=False`. Existing ACT weights initialize the base model, and the adapter
starts near zero residual.
