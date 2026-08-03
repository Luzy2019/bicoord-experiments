# FM_MR: Adaptive Refinement Flow Matching

This branch is copied from `policy/FM` for the adaptive refinement action expert experiment.
It does not use `FM_AT2` coupling/DAG scheduling or the old factorized policy.

## Short-Term V1

The default MR policy trains directly from demo actions; teacher-student distillation is
disabled unless a checkpoint is explicitly provided.

The action expert keeps the environment/data horizon dense (`horizon: 30`) but uses a
small low-level expert window (`expert_horizon: 10`). A gate is computed once per
top-level policy call and reused through the flow-matching sampling loop.

Each active arm/refinement round runs only a 10-token expert call. For example, one arm
can use one round (`10` generated action tokens) while the other uses three rounds
(`10 + 10 + 10 = 30` generated action tokens). The final per-arm trajectories are
densified back to the runner-compatible joint `action` / `action_pred`.

Current training does not require phase/subtask labels. Refine targets use a front-weighted
velocity/acceleration complexity fallback and are isolated behind a replaceable target
builder for future subtask/stage-label supervision.
