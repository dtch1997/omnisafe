# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of training a policy with OmniSafe."""

import argparse

import omnisafe
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict

import wandb
import time

def init_wandb(args, config, run_name) -> "wandb.Run":
    return wandb.init(
        name=run_name,
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        config=config,
        sync_tensorboard=True        
    )

def add_wandb_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--track",
        action="store_true",
        default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name", type=str, default="svf_gymnasium", help="the wandb's project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default='dtch1997',
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="default",
        help="the wandb's group name for the current run",
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    parser.add_argument(
        "-tags",
        "--wandb-tags",
        type=str,
        default=[],
        nargs="+",
        help="Tags for wandb run, e.g.: -tags optimized pr-123",
    )
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        metavar='ALGO',
        default='PPOLag',
        help='algorithm to train',
        choices=omnisafe.ALGORITHMS['all'],
    )
    parser.add_argument(
        '--env-id',
        type=str,
        metavar='ENV',
        default='SafetyPointGoal1-v0',
        help='the name of test environment',
    )
    parser.add_argument(
        '--parallel',
        default=1,
        type=int,
        metavar='N',
        help='number of paralleled progress for calculations.',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=10000000,
        metavar='STEPS',
        help='total number of steps to train for algorithm',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        metavar='DEVICES',
        help='device to use for training',
    )
    parser.add_argument(
        '--vector-env-nums',
        type=int,
        default=1,
        metavar='VECTOR-ENV',
        help='number of vector envs to use for training',
    )
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=16,
        metavar='THREADS',
        help='number of threads to use for torch',
    )
    args, unparsed_args = parser.parse_known_args()
    
    parser = add_wandb_args(parser)
    wandb_args, unparsed_args = parser.parse_known_args(unparsed_args)
    if wandb_args.track: 
        run_name = f"{args.env_id}__{args.algo}__{int(time.time())}"
        init_wandb(wandb_args, args, run_name)

    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))

    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        train_terminal_cfgs=vars(args),
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
