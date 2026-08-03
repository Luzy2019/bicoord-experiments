# DP_MR: Adaptive Refinement Diffusion Policy

This branch is copied from `policy/DP` for the adaptive refinement action expert experiment.
It mirrors `FM_MR` with a DDPM backbone.

## Stages

Train a same-rate teacher:

```bash
bash train_mr.sh stack_bowls demo_clean 50 42 16 0 teacher
```

Train an adaptive refinement student:

```bash
bash train_mr.sh stack_bowls demo_clean 50 42 16 0 student checkpoints/stack_bowls-demo_clean-50-42-teacher/600.ckpt
```

The student uses one transformer action expert with a shared trunk and separate left/right
action heads. Each arm has its own refinement gate, cache/reuse score, and compute budget,
then the two arm outputs are stitched back into dense `action` and `action_pred` for the
existing runner interface.

Each refinement round increases the internal action resolution. For example, with
`coarse_plan_steps: 10`, `horizon: 30`, and `max_refine_rounds: 3`, an arm can select
10, 20, or 30 generated action points over the same semantic trajectory window. Left and
right select these resolutions independently.
