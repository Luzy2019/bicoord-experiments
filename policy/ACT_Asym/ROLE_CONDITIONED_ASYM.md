# Role-Conditioned Asymmetric ACT

`policy/ACT_Asym` is an isolated copy of ACT with asymmetric bimanual execution.
The original `policy/ACT` directory is not required by this variant.

## What Is Added

The learned scheduler predicts:

```text
left_update_gate[t], right_update_gate[t]
role_prob
```

Role ids are:

```text
0 balanced
1 left_primary
2 right_primary
3 static
```

If a role is supplied at inference, it conditions the scheduler. If no role is
supplied, the scheduler infers a soft role from the predicted action chunk.

## Training With Existing Same-Frequency Data

Use the learned mode to train from current ACT datasets:

```bash
bash train_asym_bimanual.sh TASK_NAME TASK_CONFIG EXPERT_NUM SEED GPU 1 5 6000 8 "" original learned
```

This keeps the original same-frequency action target, then learns update gates
from motion-derived pseudo labels:

```text
large arm motion change -> high update probability
small arm motion change -> low update probability
```

The update gate is trained with Bernoulli KL, and the coarse role classifier is
trained with pseudo roles from left/right motion energy.

## Supplying Task Roles At Inference

The model reads either:

```python
obs["asym_role_id"] = 1
```

or:

```python
obs["asym_role"] = "left_primary"
```

Supported role strings include:

```text
balanced, left_primary, right_primary, static
left_high, right_high, left_operate, right_operate
```

`deploy_policy.py` also checks `TASK_ENV.get_asym_role()` when available.

## Executor-Level Asymmetric Dispatch

`ACT.get_action(obs, return_packet=True)` returns:

```python
{
    "action": full_14d_action,
    "left_action": left_7d_action,
    "right_action": right_7d_action,
    "left_update": bool,
    "right_update": bool,
    "role_prob": optional_role_distribution,
}
```

`deploy_policy.dispatch_asymmetric_action()` uses the best available API:

1. `TASK_ENV.take_action_asym(left_action, right_action, left_update, right_update)`
2. `TASK_ENV.take_left_action(...)` and `TASK_ENV.take_right_action(...)`
3. fallback to `TASK_ENV.take_action(full_action)`

The fallback remains compatible with existing environments, but true
communication-frequency asymmetry requires API 1 or 2.
