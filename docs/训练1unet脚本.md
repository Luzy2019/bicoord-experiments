# 1-UNet / factorized staged training examples

cd policy/DP

# 只训练 base：600 epoch，每 50 epoch 保存一次
# 训练参数先修改：policy/DP/diffusion_policy/config/robot_dp_14.yaml 的 factorized_stages.base_stage1
# 输出目录：checkpoints/stack_bowls-demo_clean-50-100
bash train_factorized_base_stage1.sh robot_dp_14

# 只训练 speed：从 base 的 400.ckpt 初始化，100 epoch，每 5 epoch 保存一次
# 训练参数先修改：policy/DP/diffusion_policy/config/robot_dp_14.yaml 的 factorized_stages.speed_stage2
# 读取：training.init_ckpt_path
# 输出目录：checkpoints/stack_bowls-demo_clean-50-100-speed
bash train_factorized_speed_stage2.sh robot_dp_14

# eval base：评估 base 的 400.ckpt，关闭 speed modulation
bash eval_trained.sh stack_bowls demo_clean demo_clean 50 100 0 400 false

# eval speed：评估 speed 的 100.ckpt，读取 checkpoints/stack_bowls-demo_clean-50-100
bash eval_trained.sh stack_bowls demo_clean demo_clean 50 100 0 400 false none 60 \
    stack_bowls-demo_clean-50-100

# eval speed 并保存 factorized debug 信息
bash eval_trained.sh stack_bowls demo_clean demo_clean 50 100 0 100 true debug_factorized 60 \
    stack_bowls-demo_clean-50-100