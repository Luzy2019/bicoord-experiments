# 2-UNet staged training examples

cd policy/DP

# stage 1：训练 base，600 epoch，每 50 epoch 保存一次
# 输出目录：checkpoints/stack_bowls-demo_clean-50-100
bash train_2unet_base_stage1.sh stack_bowls demo_clean 50 100 14 0 \
    160 0.25 -2.0 50 600

# stage 2：训练 speed，从 base 的 600.ckpt 初始化，100 epoch，每 5 epoch 保存一次
# 读取：checkpoints/stack_bowls-demo_clean-50-100/600.ckpt
# 输出目录：checkpoints/stack_bowls-demo_clean-50-100-speed
bash train_2unet_speed_stage2.sh stack_bowls demo_clean 50 100 14 0 \
    checkpoints/stack_bowls-demo_clean-50-100/500.ckpt \
    160 1e-2 5 100

# eval base：评估 base 的 600.ckpt，关闭 speed modulation
bash eval_trained.sh stack_bowls demo_clean demo_clean 50 100 0 600 false

# eval speed：评估 speed 的 100.ckpt，读取 checkpoints/stack_bowls-demo_clean-50-100-speed
bash eval_trained.sh stack_bowls demo_clean demo_clean 50 100 0 100 true none 60 \
    stack_bowls-demo_clean-50-100-speed

---
cd policy/DP

# eval base，关闭 speed modulation
bash eval_trained.sh stack_bowls demo_clean demo_clean 50 100 0 600 false

# eval speed，读取 speed stage 输出目录
bash eval_trained.sh stack_bowls demo_clean demo_clean 50 100 0 100 true none 60 stack_bowls-demo_clean-50-100

# eval speed，并保存 factorized debug
bash eval_trained.sh stack_bowls demo_clean demo_clean 50 100 0 100 true debug_factorized 60 stack_bowls-demo_clean-50-100