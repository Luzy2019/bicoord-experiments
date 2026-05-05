# Asymmetric Bimanual Diffusion Policy

This policy is copied from `policy/DP` and implements the design in
`demos/chatgpt_5.5_thinking.md` and `demos/claude_4.7_opus.md`.

## Model

The original DP trains one joint diffusion noise predictor:

```text
epsilon = U-Net(z_t, t, obs)
```

`DP_Asym` uses a role-adaptive separated-arm variant. The left and right arms
are diffused as two internal sequences:

```text
z_left  : [B, ceil(H / left_stride),  D_left]
z_right : [B, ceil(H / right_stride), D_right]
```

Strides are kept as ablation knobs, but the default is now full-rate for both
arms (`left_stride=1`, `right_stride=1`). For example, with `H=8`,
`left_stride=1`, `right_stride=4`:

```text
left  generates 8 internal actions: t0 t1 t2 t3 t4 t5 t6 t7
right generates 2 internal actions: t0             t4
```

The right arm is then expanded by holding its latest generated value until the
next right-arm update. This means the two arms are not merely generated at the
same frequency and masked afterward; they are different diffusion variables.

Inside those separated variables, `DP_Asym` uses a cost-aware mixture of four
bimanual structures plus a dynamic support-role head:

```text
epsilon =
  pi_A  * epsilon_A
+ pi_LR * epsilon_LR
+ pi_RL * epsilon_RL
+ pi_C  * epsilon_C
```

The experts mean:

```text
A  : joint bimanual expert
LR : left leader, right follower
RL : right leader, left follower
C  : independent local arms
```

The gating network predicts `pi_A, pi_LR, pi_RL, pi_C` and
`support_prob=[left_support, right_support]` from the expanded noisy left/right
action state, diffusion step, and encoded observation. The support probability
lets the model decide which arm should be smoother/stabler for the current
state instead of hard-coding "left fine, right stable" or the reverse.

The cost order is:

```text
cost(A) > cost(LR) = cost(RL) > cost(C)
```

This makes the model prefer independent execution unless the diffusion loss
requires stronger cross-arm coupling.

## Training Objective

The total loss is:

```text
L = L_DDPM
  + lambda_cost          * E[pi_A cost_A + pi_LR cost_B + pi_RL cost_B + pi_C cost_C]
  + lambda_entropy       * H(pi)
  + lambda_update_sparse * fixed_update_rate(left_stride, right_stride)
  + lambda_smooth        * ||epsilon[t+1] - epsilon[t]||^2
  + lambda_role_smooth   * support_weighted_arm_smoothness
```

Training reuses synchronous demonstrations by downsampling each arm according to
its own stride before adding diffusion noise. The full-rate target is therefore
not learned first and projected later; the left and right diffusion targets are
strictly separate.

## Train

From `policy/DP_Asym`:

```bash
bash train_asym_bimanual.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id>

```

Example:

```bash
bash train_asym_bimanual.sh pick_place demo 100 42 14 0
bash train_asym_bimanual.sh handover_block_with_bowls demo_clean 50 42 14 0
```

Optional role-adaptive objective weights:

```bash
bash train_asym_bimanual.sh pick_place demo 100 42 14 0 \
  1 1 1e-3 1e-4 1e-3 0.0 1e-3
```

The first two optional values are `left_stride` and `right_stride`. Keep both
as `1` for the default dynamic role-adaptive policy. The last value is
`lambda_role_smooth`, which weights the support-role smoothness loss.

The shorter `train.sh` is also updated to use the asymmetric config.

## Inference

Use the normal project evaluator with the new policy name:

```bash
bash eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id>
```

The deployment entry points are:

```text
policy/DP_Asym/deploy_policy.py
policy/DP_Asym/deploy_policy_double_env.py
policy/DP_Asym/dp_model.py
```

At inference, the policy samples the compressed left and right diffusion
trajectories separately, then expands them to the environment action rate with
the configured hold rule.

## Main Config

The 14-DoF and 16-DoF configs are:

```text
diffusion_policy/config/robot_dp_14.yaml
diffusion_policy/config/robot_dp_16.yaml
```

Important knobs:

```text
policy.asym_cost_weights
policy.asym_lambda_cost
policy.asym_lambda_entropy
policy.asym_lambda_update_sparse
policy.asym_lambda_smooth
policy.left_stride
policy.right_stride
policy.asym_lambda_role_smooth
```
