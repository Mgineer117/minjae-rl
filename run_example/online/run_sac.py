import sys
sys.path.append('../minjae-rl')

import uuid
import argparse
import random
import os

import numpy as np
import torch

from rlkit.utils.utils import seed_all
from rlkit.nets import MLP, RNNModel, RecurrentEncoder
from rlkit.modules import ActorProb, Critic, DistCritic, PhiNetwork, TanhDiagGaussian
from rlkit.utils.load_dataset import qlearning_dataset
from rlkit.utils.load_env import load_env
from rlkit.utils.zfilter import ZFilter
from rlkit.buffer import OnlineSampler
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy_trainer import MFPolicyTrainer
from rlkit.policy import SACPolicy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="popodice")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument('--task', type=str, default=None) # None for Gym and MetaGym except ML1 or MT1
    parser.add_argument("--algo-name", type=str, default="sac")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="log")

    parser.add_argument('--env-type', type=str, default='MetaGym') # Gym or MetaGym
    parser.add_argument('--agent-type', type=str, default='MT1') # MT1, ML45, Hopper, Ant
    parser.add_argument('--task-name', type=str, default='pick-place') # None for Gym and MetaGym except ML1 or MT1
    parser.add_argument('--task-num', type=int, default=3) # 10, 45, 50

    parser.add_argument('--seeds', default=[1, 3, 5, 7, 9], type=list)
    parser.add_argument('--pklfile', type=str, default=None)
    parser.add_argument('--actor-hidden-dims', default=(256, 256))
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=10)
    parser.add_argument('--episode_len', type=int, default=500)
    parser.add_argument('--episode_num', type=int, default=2)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

def train(args=get_args()):
    unique_id = str(uuid.uuid4())[:4]
    
    for seed in args.seeds:
        # create env and dataset
        args.task = '-'.join((args.env_type, args.agent_type))
        training_envs, testing_envs = load_env(args.task, args.task_name, args.task_num)

        # seed
        seed_all(seed)

        # create policy model
        '''state dimension input manipulation for Ss only'''
        args.obs_shape = (training_envs[0].observation_space.shape[0],)
        args.action_dim = np.prod(training_envs[0].action_space.shape)
        args.max_action = training_envs[0].action_space.high[0]
        
        running_state = ZFilter(args.obs_shape, clip=5)

        sampler = OnlineSampler(
            obs_shape=args.obs_shape,
            action_dim=args.action_dim,
            episode_len= args.episode_len,
            episode_num= args.episode_num,
            running_state=running_state,
            device=args.device,
        )

        actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.actor_hidden_dims)
        critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, activation=torch.nn.Tanh, hidden_dims=args.hidden_dims)
        critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        
        dist = TanhDiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=args.action_dim,
            unbounded=True,
            conditioned_sigma=True,
            max_mu=args.max_action
        )
        
        actor = ActorProb(actor_backbone,
                          dist,
                          device = args.device)   
        
        critic1 = Critic(critic1_backbone, device = args.device)
        critic2 = Critic(critic2_backbone, device = args.device)

        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

        policy = SACPolicy(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim
        )

        # setup logger
        default_cfg = vars(args)#asdict(args)
        if args.name is None:
            args.name = args.algo_name + '-' + unique_id + "-seed" + str(seed)
        if args.group is None:
            args.group = args.task + "-seed-" + str(seed)
        if args.logdir is not None:
            args.logdir = os.path.join(args.logdir, args.name, args.group)  
        logger = WandbLogger(default_cfg, args.project, args.group, args.name, args.logdir)
        logger.save_config(default_cfg, verbose=args.verbose)

        # create policy trainer
        policy_trainer = MFPolicyTrainer(
            policy=policy,
            train_env=training_envs,
            eval_env=testing_envs,
            sampler=sampler,
            logger=logger,
            epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            eval_episodes=args.eval_episodes,
            obs_dim=args.obs_shape,
            device=args.device,
        )

        # train
        policy_trainer.online_train(seed)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()