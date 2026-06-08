# FM_MR: Direct Multi-Rate Flow Matching

This branch is copied from `policy/FM` for the multi-rate action chunking experiment.
It does not use `FM_AT2` coupling/DAG scheduling or the old factorized policy.

## Stages

Train a same-rate teacher:

```bash
bash train_mr.sh stack_bowls demo_clean 50 100 16 0 teacher
```

Train a direct multi-rate student:

```bash
bash train_mr.sh stack_bowls demo_clean 50 100 16 0 student checkpoints/stack_bowls-demo_clean-50-100-teacher/600.ckpt
```

The student directly predicts:

```text
main_chunk:   [B, H_main, D_main]
assist_chunk: [B, H_assist, D_assist]
```

and unrolls them into a dense joint `action_pred` for the existing environment interface.
