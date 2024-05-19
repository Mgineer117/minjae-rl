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
from rlkit.modules import DiceActor, Critic, TanhDiagGaussian, TanhMixDiagGaussian
from rlkit.utils.load_dataset import qlearning_dataset
from rlkit.buffer import ReplayBuffer
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy_trainer import MFPolicyTrainer
from rlkit.policy import OPDPolicy

POLICY_EXTRACTION = ['wbc', 'iproj']
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
    'hopper': [0, 5, 6, 8, 9, 10], 
    'walker2d': [8, 9, 10, 11, 12, 13, 14, 15, 16],
    'halfcheetah': [4, 5, 6, 7, 13, 14, 15, 16]
}
DATA_POLICY = ['tanh_normal', 'tanh_mdn']
F = ['chisquare', 'kl', 'elu']
GENDICE_LOSS_TYPE = ['gendice', 'bestdice']
E_LOSS_TYPE = ['mse', 'minimax']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="popodice")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--algo-name", type=str, default="optidice")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument('--seeds', default=[1, 3, 5, 7, 9], type=list)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)                  ########## TUNE
    parser.add_argument('--actor-hidden-dims', default=(256, 256))
    parser.add_argument('--hidden-dims', default=(256, 256))

    parser.add_argument('--policy_extraction', default='iproj', type=str, choices=POLICY_EXTRACTION)
    parser.add_argument('--log_iterations', default=int(1), type=int)
    parser.add_argument('--data_policy', default='tanh_mdn', type=str, choices=DATA_POLICY)
    parser.add_argument('--data_policy_num_mdn_components', default=1, type=int)
    parser.add_argument('--data_policy_mdn_temperature', default=1.0, type=float)
    parser.add_argument('--mean_range', default=(-7.24, 7.24))
    parser.add_argument('--logstd_range', default=(-5., 2.))
    parser.add_argument('--data_mean_range', default=(-7.24, 7.24))
    parser.add_argument('--data_logstd_range', default=(-5., 2.))
    parser.add_argument('--use_policy_entropy_constraint', default=True, type=bool)          ########## TUNE
    parser.add_argument('--use_data_policy_entropy_constraint', default=False, type=bool)          ########## TUNE
    parser.add_argument('--target_entropy', default=None, type=float)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--alpha', default=1.0, type=float)                  ###    ####### TUNE
    parser.add_argument('--f', default='elu', type=str, choices=F) 
    parser.add_argument('--gendice_v', default=True, type=bool)
    parser.add_argument('--gendice_e', default=True, type=bool)
    parser.add_argument('--gendice_loss_type', default='bestdice', type=str, choices=GENDICE_LOSS_TYPE)
    parser.add_argument('--normalize-obs', default=True, type=bool)
    parser.add_argument('--normalize-rewards', default=True, type=bool)
    parser.add_argument('--reward_scale', default=1e-1, type=float)
    parser.add_argument('--e_loss_type', default='minimax', type=str, choices=E_LOSS_TYPE)          ########## TUNE
    parser.add_argument('--v_l2_reg', default=None, type=float)
    parser.add_argument('--e_l2_reg', default=None, type=float)
    parser.add_argument('--lamb_scale', default=1.0, type=float)          ########## TUNE
    
    parser.add_argument("--epoch", type=int, default=1000)
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
        dataset = d4rl.qlearning_dataset(env)
        args.obs_shape = (env.observation_space.shape[0] - len(pomdp),) if pomdp is not None else env.observation_space.shape
        args.action_dim = np.prod(env.action_space.shape)
        args.max_action = env.action_space.high[0]
        print('     max action:  ', args.max_action)
        args.target_entropy = -args.action_dim # proposed by SAC (Haarnoja et al., 2018) (−dim(A) for each task).

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
            
        if args.normalize_obs:
            _, _ = buffer.normalize_obs()

        if args.normalize_rewards:
            _,_ = buffer.normalize_rewards()

        if args.pomdp:
            buffer.make_pomdp()

        # seed
        args.seed = seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

        # create policy model
        actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.actor_hidden_dims, mlp_initialization=True)
        data_actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.actor_hidden_dims, mlp_initialization=True)
        v_network_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, mlp_initialization=True)
        e_network_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims, mlp_initialization=True)

        actor = DiceActor(actor_backbone,
                          latent_dim=getattr(actor_backbone, "output_dim"),
                          output_dim=args.action_dim,
                          max_action = args.max_action,
                          device = args.device)
        
        data_actor = DiceActor(data_actor_backbone,
                    latent_dim=getattr(data_actor_backbone, "output_dim"),
                    output_dim=args.action_dim,
                    max_action = args.max_action,
                    device = args.device)
        
        v_network = Critic(v_network_backbone, args.device)
        e_network = Critic(e_network_backbone, args.device)
        v_network_optim = torch.optim.Adam(v_network.parameters(), lr=args.critic_lr)
        e_network_optim = torch.optim.Adam(e_network.parameters(), lr=args.critic_lr)

        # create policy
        policy = OPDPolicy(
            actor,
            data_actor,
            v_network,
            e_network,
            v_network_optim,
            e_network_optim,
            args)

        # setup logger
        default_cfg = vars(args)#asdict(args)
        if args.name is None:
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
            obs_mean = buffer._obs_mean,
            obs_std = buffer._obs_std,
            pomdp=pomdp
        )

        # train
        policy_trainer.train(seed, normalize=args.normalize_obs)


if __name__ == "__main__":
    testing_param = [[1, 3, 5, 7, 9]]
    for param in testing_param:
        train(param=param)