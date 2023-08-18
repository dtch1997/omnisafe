
#!/bin/bash

group_name=${1:-'safety-violation-expt'}

for seed in 0 1 2
do 
    for task in Goal # Circle Push
    do 
        for robot in Point Car # Doggo
        do
            for level in 1 2
            do 
                for algo in PPO PPOLag SAC SACLag FOCOPS CPO 
                do
                    env_id="Safety${robot}${task}${level}-v0"
                    python examples/train_policy.py \
                        --env-id $env \
                        --algo $algo \
                        --track \
                        --total-steps 1000000 \
                        --wandb-group $group_name \
                        --seed $seed
                done
            done
        done
    done
done