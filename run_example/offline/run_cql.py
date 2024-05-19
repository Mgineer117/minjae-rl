import sys
sys.path.append('../minjae-rl')

import uuid
import argparse
import random
import os

import gym
import d4rl
import wandb

import numpy as np
import torch


from rlkit.nets import MLP
from rlkit.modules import ActorProb, Critic, TanhDiagGaussian
from rlkit.utils.load_dataset import qlearning_dataset
from rlkit.buffer import ReplayBuffer
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy_trainer import MFPolicyTrainer
from rlkit.policy import CQLPolicy

ENV_NAME = [
    # Gym
    'halfcheetah-random-v0',
    'walker2d-random-v0',
    'hopper-random-v0',
    'halfcheetah-medium-v0',
    'walker2d-medium-v0',
    'hopper-medium-v0',
    'halfcheetah-medium-replay-v0',
    'walker2d-medium-replay-v0',
    'hopper-medium-replay-v0',
    'halfcheetah-medium-expert-v0',
    'walker2d-medium-expert-v0',
    'hopper-medium-expert-v0',
]
MASKING_IDX = {
    # Index to delete
    'hopper': [5, 6], 
    'walker2d': [8, 9],
    'halfcheetah': [4, 5]
}

"""
suggested hypers
cql-weight=5.0, temperature=1.0 for all D4RL-Gym tasks
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="popodice")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--algo-name", type=str, default="cql")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument('--seeds', default=[1, 3, 5, 7, 9], type=list)
    parser.add_argument('--actor-hidden-dims', default=(64, 64))
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--pomdp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(param, args=get_args()):
    random_uuid = str(uuid.uuid4())[:4]
    args.seeds = param

    domain_name = args.task.split('-')[0]
    pomdp = MASKING_IDX[domain_name] if args.pomdp == True else None
    for seed in args.seeds:    
        # create env and dataset
        env = gym.make(args.task)
        dataset = qlearning_dataset(env)
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        if 'antmaze' in args.task:
            dataset["rewards"] = (dataset["rewards"] - 0.5) * 4.0
        args.obs_shape = (env.observation_space.shape[0] - len(pomdp),) if pomdp is not None else env.observation_space.shape
        args.action_dim = np.prod(env.action_space.shape)
        args.max_action = env.action_space.high[0]

        # seed
        args.seed = seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

        # create policy model
        actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.actor_hidden_dims)
        critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        dist = TanhDiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=args.action_dim,
            unbounded=True,
            conditioned_sigma=True,
            max_mu=args.max_action
        )
        actor = ActorProb(actor_backbone, dist, args.device)
        critic1 = Critic(critic1_backbone, args.device)
        critic2 = Critic(critic2_backbone, args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

        if args.auto_alpha:
            target_entropy = args.target_entropy if args.target_entropy \
                else -np.prod(env.action_space.shape)

            args.target_entropy = target_entropy

            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)
        else:
            alpha = args.alpha

        # create policy
        policy = CQLPolicy(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space=env.action_space,
            tau=args.tau,
            gamma=args.gamma,
            alpha=alpha,
            cql_weight=args.cql_weight,
            temperature=args.temperature,
            max_q_backup=args.max_q_backup,
            deterministic_backup=args.deterministic_backup,
            with_lagrange=args.with_lagrange,
            lagrange_threshold=args.lagrange_threshold,
            cql_alpha_lr=args.cql_alpha_lr,
            num_repeart_actions=args.num_repeat_actions
        )

        domain_name = args.task.split('-')[0]
        pomdp = MASKING_IDX[domain_name] if args.pomdp == True else None

        # create buffer
        buffer = ReplayBuffer(
            buffer_size=len(dataset["observations"]),
            obs_shape=args.obs_shape,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32,
            pomdp=pomdp,
            device=args.device
        )
        buffer.load_dataset(dataset)

        # setup logger
        default_cfg = vars(args)#asdict(args)
        if args.name is None:
            random_uuid = str(uuid.uuid4())[:4]
            args.name = args.algo_name + '-' + random_uuid + "-seed" + str(seed)
        if args.group is None:
            args.group = args.task + "-seed-" + str(seed)
        if args.logdir is not None:
            args.logdir = os.path.join(args.logdir, args.name, args.group)  
        logger = WandbLogger(default_cfg, args.project, args.group, args.name, args.logdir)
        logger.save_config(default_cfg, verbose=args.verbose)

        # create policy trainer
        policy_trainer = MFPolicyTrainer(
            policy=policy,
            eval_env=env,
            buffer=buffer,
            logger=logger,
            epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            batch_size=args.batch_size,
            eval_episodes=args.eval_episodes,
            pomdp=pomdp
        )

        # train
        policy_trainer.train(seed)


if __name__ == "__main__":
    testing_param = [[1, 3, 5, 7, 9]]
    for param in testing_param:
        train(param=param)