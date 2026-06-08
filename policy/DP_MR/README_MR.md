# DP_MR: Direct Multi-Rate Diffusion Policy

This branch is copied from `policy/DP` for the multi-rate action chunking experiment.
It mirrors `FM_MR` with a DDPM backbone.

## Stages

Train a same-rate teacher:

```bash
bash train_mr.sh stack_bowls demo_clean 50 42 16 0 teacher
```

Train a direct multi-rate student:

```bash
bash train_mr.sh stack_bowls demo_clean 50 42 16 0 student checkpoints/stack_bowls-demo_clean-50-42-teacher/600.ckpt
```

The student directly predicts compact main/assist chunks and returns a dense `action_pred`
for compatibility with existing runners.
