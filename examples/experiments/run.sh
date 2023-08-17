
#!/bin/bash

# The variable $1 refers to the first argument passed to the script. 
# If no argument is passed, 'Default Value' will be used.
seed=${1:-'0'}
group_name=${2:-'sac_baselines'}

for env in SafetyPointGoal1-v0 SafetyDoggoGoal1-v0 SafetyCarGoal1-v0
do 
    for algo in SAC SACLag
    do
        python examples/train_policy.py \
            --env-id $env \
            --algo $algo \
            --track \
            --total-steps 1000000 \
            --group-name $group_name
    done
done