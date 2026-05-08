# stage 1
bash train_factorized_speed_staged.sh stack_bowls demo_clean 50 100 14 0 96 0.25 -2.0 1e-2 auto true

# stage 2
bash train_factorized_speed_staged.sh stack_bowls demo_clean 50 100 14 0 96 0.25 -2.0 1e-2 400 false
