import sys
sys.path.append('../minjae-rl')

import uuid
import argparse
import random
import os

import numpy as np
import torch

from rlkit.utils.utils import seed_all, select_device
from rlkit.nets import MLP
from rlkit.modules import ActorProb, Critic, DiagGaussian
from rlkit.utils.load_dataset import qlearning_dataset
from rlkit.utils.load_env import load_env
from rlkit.utils.zfilter import ZFilter
from rlkit.buffer import OnlineSampler
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy_trainer import MFPolicyTrainer
from rlkit.policy import TRPOPolicy

def get_args():
    parser = argparse.ArgumentParser()
    '''WandB and Logging parameters'''
    parser.add_argument("--project", type=str, default="data-collect")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument('--task', type=str, default=None) # None, naming began using environmental parameters
    parser.add_argument("--algo-name", type=str, default="trpo")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="log")

    '''OpenAI Gym parameters'''
    parser.add_argument('--env-type', type=str, default='Gym') # Gym or MetaGym
    parser.add_argument('--agent-type', type=str, default='Hopper') # MT1, ML45, Hopper, Ant
    parser.add_argument('--task-name', type=str, default=None) # None for Gym and MetaGym except ML1 or MT1 'pick-place'
    parser.add_argument('--task-num', type=int, default=1) # 10, 45, 50

    '''Algorithmic and sampling parameters'''
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num-cores', type=int, default=None)
    parser.add_argument('--actor-hidden-dims', default=(256, 256))
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=50)
    parser.add_argument('--episode_len', type=int, default=1000)
    parser.add_argument('--episode_num', type=int, default=2)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--grad-norm", type=bool, default=False)
    parser.add_argument("--rendering", type=bool, default=False)
    parser.add_argument("--data_num", type=int, default=1_000_000)
    parser.add_argument("--import-policy", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()

def train(args=get_args()):
    unique_id = str(uuid.uuid4())[:4]
    args.device = select_device()
    
    # seed
    seed_all(args.seed)

    # create env and dataset
    args.task = '-'.join((args.env_type, args.agent_type))
    training_envs, testing_envs, eval_env_idx = load_env(args.task, args.task_name, args.task_num, render_mode='rgb_array')

    # create policy model
    '''state dimension input manipulation for Ss only'''
    args.obs_shape = (training_envs[0].observation_space.shape[0],)
    args.action_dim = np.prod(training_envs[0].action_space.shape)
    args.max_action = training_envs[0].action_space.high[0]

    running_state = ZFilter(args.obs_shape, clip=5)

    sampler = OnlineSampler(
        obs_shape=args.obs_shape,
        action_dim=args.action_dim,
        episode_len=args.episode_len,
        episode_num=args.episode_num,
        training_envs=training_envs,
        #running_state=running_state,
        data_num=args.data_num,
        num_cores=args.num_cores,
        device=args.device,
    )

    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.actor_hidden_dims, activation=torch.nn.Tanh,)
    critic_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, activation=torch.nn.Tanh,)
    
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=False,
        conditioned_sigma=True,
        max_mu=args.max_action,
        sigma_min=-5.0,
        sigma_max=2.0
    )

    actor = ActorProb(actor_backbone,
                        dist_net=dist,
                        device=args.device)   
            
    critic = Critic(critic_backbone, device = args.device)
    #critic_optim = torch.optim.LBFGS(critic.parameters(), lr=args.critic_lr, max_iter=20)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    
    policy = TRPOPolicy(
        actor=actor,
        critic=critic,
        critic_optim=critic_optim,
        grad_norm=args.grad_norm,
        device=args.device
    )

    # setup logger
    default_cfg = vars(args)#asdict(args)
    if args.name is None:
        args.name = args.algo_name + '-' + unique_id + "-seed" + str(args.seed)
    if args.group is None:
        args.group = args.task + "-seed-" + str(args.seed)
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.name, args.group)  
    logger = WandbLogger(default_cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(default_cfg, verbose=args.verbose)

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=testing_envs,
        eval_env_idx=eval_env_idx,
        sampler=sampler,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        eval_episodes=args.eval_episodes,
        rendering=args.rendering,
        obs_dim=args.obs_shape[0],
        action_dim=args.action_dim,
        import_policy=args.import_policy,
        device=args.device,
    )

    # train
    policy_trainer.online_train(args.seed)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train()