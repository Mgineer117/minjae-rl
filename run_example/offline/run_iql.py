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
from rlkit.modules import ActorProb, Critic, DiagGaussian
from rlkit.utils.load_dataset import qlearning_dataset
from rlkit.buffer import ReplayBuffer
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy_trainer import MFPolicyTrainer
from rlkit.policy import IQLPolicy

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
expectile=0.7, temperature=3.0 for all D4RL-Gym tasks
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="popodice")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--algo-name", type=str, default="iql")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument('--seeds', default=[1, 3, 5, 7, 9], type=list)
    parser.add_argument('--actor-hidden-dims', default=(64, 64))
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-q-lr", type=float, default=3e-4)
    parser.add_argument("--critic-v-lr", type=float, default=3e-4)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--lr-decay", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--pomdp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def normalize_rewards(dataset):
    terminals_float = np.zeros_like(dataset["rewards"])
    for i in range(len(terminals_float) - 1):
        if np.linalg.norm(dataset["observations"][i + 1] -
                            dataset["next_observations"][i]
                            ) > 1e-6 or dataset["terminals"][i] == 1.0:
            terminals_float[i] = 1
        else:
            terminals_float[i] = 0

    terminals_float[-1] = 1

    # split_into_trajectories
    trajs = [[]]
    for i in range(len(dataset["observations"])):
        trajs[-1].append((dataset["observations"][i], dataset["actions"][i], dataset["rewards"][i], 1.0-dataset["terminals"][i],
                        terminals_float[i], dataset["next_observations"][i]))
        if terminals_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])
    
    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    # normalize rewards
    dataset["rewards"] /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset["rewards"] *= 1000.0

    return dataset


def train(param, args=get_args()):
    random_uuid = str(uuid.uuid4())[:4]
    args.seeds = param

    domain_name = args.task.split('-')[0]
    pomdp = MASKING_IDX[domain_name] if args.pomdp == True else None

    for seed in args.seeds:    
        # create env and dataset
        env = gym.make(args.task)
        dataset = qlearning_dataset(env)
        if 'antmaze' in args.task:
            dataset["rewards"] -= 1.0
        if ("halfcheetah" in args.task or "walker2d" in args.task or "hopper" in args.task):
            dataset = normalize_rewards(dataset)
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
        actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.actor_hidden_dims, dropout_rate=args.dropout_rate)
        critic_q1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
        critic_q2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
        critic_v_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
        dist = DiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=args.action_dim,
            unbounded=False,
            conditioned_sigma=False,
            max_mu=args.max_action
        )
        actor = ActorProb(actor_backbone, dist, args.device)
        critic_q1 = Critic(critic_q1_backbone, args.device)
        critic_q2 = Critic(critic_q2_backbone, args.device)
        critic_v = Critic(critic_v_backbone, args.device)
        
        for m in list(actor.modules()) + list(critic_q1.modules()) + list(critic_q2.modules()) + list(critic_v.modules()):
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic_q1_optim = torch.optim.Adam(critic_q1.parameters(), lr=args.critic_q_lr)
        critic_q2_optim = torch.optim.Adam(critic_q2.parameters(), lr=args.critic_q_lr)
        critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.critic_v_lr)

        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
        else:
            lr_scheduler = None
        
        # create IQL policy
        policy = IQLPolicy(
            actor,
            critic_q1,
            critic_q2,
            critic_v,
            actor_optim,
            critic_q1_optim,
            critic_q2_optim,
            critic_v_optim,
            action_space=env.action_space,
            tau=args.tau,
            gamma=args.gamma,
            expectile=args.expectile,
            temperature=args.temperature
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
            lr_scheduler=lr_scheduler,
            pomdp=pomdp
        )

        # train
        policy_trainer.train(seed)


if __name__ == "__main__":
    testing_param = [[1, 3, 5, 7, 9]]
    for param in testing_param:
        train(param=param)