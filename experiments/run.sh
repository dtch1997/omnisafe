#!/bin/bash

WANDB_PROJECT="omnisafe"
WANDB_ENTITY="dtch1997"
WANDB_RUN_GROUP="saferl_mujoco"

# CPO experiments
for seed in 1 2 3
do
    for env_id in Hopper-v4 Walker2d-v4 Ant-v4
    do
            WANDB_NAME=CPO-$env_id-seed=$seed omnisafe train \
            --env-id $env_id \
            --algo CPO \
            --custom-cfgs seed --custom-cfgs $seed \
            --custom-cfgs algo_cfgs:cost_limit --custom-cfgs 0 \
            --custom-cfgs train_cfgs:total_steps --custom-cfgs 1000000
    done
done

# PPOLag experiments
for seed in 1 2 3
do
    for env_id in Hopper-v4 Walker2d-v4 Ant-v4
    do
            WANDB_NAME=PPOLag-$env_id-seed=$seed omnisafe train \
            --env-id $env_id \
            --algo PPOLag \
            --custom-cfgs seed --custom-cfgs $seed \
            --custom-cfgs lagrange_cfgs:cost_limit --custom-cfgs 0 \
            --custom-cfgs train_cfgs:total_steps --custom-cfgs 1000000
    done
done