# stage 1
bash train_2unet_base.sh stack_bowls demo_clean 50 100 14 0 160 0.25 -2.0

# stage 2
bash train_2unet_speed_stage2.sh stack_bowls demo_clean 50 100 14 0 \
    checkpoints/stack_bowls-demo_clean-50-100-factorized_base/400.ckpt \
    160 1e-2 600