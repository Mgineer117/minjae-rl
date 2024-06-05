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

from rlkit.utils.utils import seed_all, select_device
from rlkit.nets import MLP, RNNModel, RecurrentEncoder, BaseEncoder
from rlkit.modules import DiceActor, Critic, TanhDiagGaussian, TanhMixDiagGaussian
from rlkit.utils.load_dataset import qlearning_dataset
from rlkit.utils.load_buffer import collect_gym_buffers, collect_metagym_buffers, collect_d4rl_buffers

from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy_trainer import MFMAMLPolicyTrainer
from rlkit.policy import OPTMAMLPolicy

POLICY_EXTRACTION = ['wbc', 'iproj']
GYM_ENV_NAME = [
    # Gym
    'HalfCheetah',
    'Walker2d',
    'Hopper',
]

F = ['chisquare', 'kl', 'elu']
GENDICE_LOSS_TYPE = ['gendice', 'bestdice']
E_LOSS_TYPE = ['mse', 'minimax']

def get_args():
    parser = argparse.ArgumentParser()
    '''WandB and Logging parameters'''
    parser.add_argument("--project", type=str, default="optimaml")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--algo-name", type=str, default="opti-maml")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="log")

    '''Env parameters'''
    parser.add_argument('--env-type', type=str, default='MetaGym') # Gym or MetaGym or d4rl
    parser.add_argument('--agent-type', type=str, default='ML10') # MT1, ML45, Hopper, Ant
    parser.add_argument('--task-name', type=str, default=None) # None for Gym and MetaGym except ML1 or MT1 'pick-place'
    parser.add_argument('--task-num', type=int, default=None) # only for Gym: 2 3 4 5

    '''Algorithmic parameters'''
    parser.add_argument('--seeds', default=[1, 3, 5, 7, 9], type=list)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)                  ########## TUNE
    parser.add_argument('--actor-hidden-dims', default=(256, 256))
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument('--policy_extraction', default='wbc', type=str, choices=POLICY_EXTRACTION)
    parser.add_argument('--log_iterations', default=int(1), type=int)
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
    parser.add_argument("--local-steps", type=int, default=3)
    parser.add_argument("--embed-type", type=str, default='skill') # skill, task, or none
    parser.add_argument("--embed-dim", type=int, default=5)

    '''Sampling parameters'''
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-trj", type=int, default=10)
    parser.add_argument("--rendering", type=bool, default=True)
    parser.add_argument("--gpu-idx", type=int, default=0)
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()

def train(args=get_args()):
    random_uuid = str(uuid.uuid4())[:4]
    args.device = select_device(args.gpu_idx)
    
    for seed in args.seeds:
        # seed
        seed_all(seed)

        # create env for evaluation purpose
        args.task = '-'.join((args.env_type, args.agent_type))
        
        if args.env_type =='Gym':
            # For Gym, we use multiple reward dynamics for multiple tasks
            training_buffers, testing_buffer, eval_env = collect_gym_buffers(args)
        elif args.env_type =='MetaGym':
            training_buffers, testing_buffer, eval_env = collect_metagym_buffers(args)
        elif args.env_type == 'd4rl':
            training_buffers, testing_buffer, eval_env = collect_d4rl_buffers(args)
        else:
            NotImplementedError

        args.obs_shape = (eval_env.observation_space.shape[0], )
        args.action_dim = np.prod(eval_env.action_space.shape)
        args.max_action = eval_env.action_space.high[0]
        args.target_entropy = -args.action_dim # proposed by SAC (Haarnoja et al., 2018) (−dim(A) for each task).

        # define encoder 
        if args.embed_type =='skill':
            rnn_size = int(np.prod(args.obs_shape) + args.action_dim + np.prod(args.obs_shape) + 1)
            encoder = RecurrentEncoder(
                input_size=rnn_size, 
                hidden_size=rnn_size, 
                output_size=args.embed_dim,
                output_activation=torch.nn.Tanh(),
                device = args.device
            )
            encoder_optim = torch.optim.Adam(encoder.parameters(), lr=args.critic_lr)
            masking_indices = [0, 2, 3, 4, 7, 8, 9, 10] #[0, 5, 6, 7, 8, 9, 10] #[0, 1, 2, 3, 4, 5, 6]
            masking_indices_length = len(masking_indices)
            if args.num_trj == 0:
                args.num_trj = 5
                print('Warning: num_trj is set to 0; recurrent encoder requires it to be a trj input')
                print('setting in to 5')
        elif args.embed_type == 'task':
            rnn_size = int(np.prod(args.obs_shape) + args.action_dim + np.prod(args.obs_shape) + 1)
            encoder = RecurrentEncoder(
                input_size=rnn_size, 
                hidden_size=rnn_size, 
                output_size=args.embed_dim,
                output_activation=torch.nn.Tanh(),
                device = args.device
            )
            encoder_optim = torch.optim.Adam(encoder.parameters(), lr=args.critic_lr)
            masking_indices = None
            masking_indices_length = 0
            if args.num_trj == 0:
                args.num_trj = 5
                print('Warning: num_trj is set to 0; recurrent encoder requires it to be a trj input')
                print('setting in to 5')
        else:
            encoder = BaseEncoder(device=args.device)
            encoder_optim = None
            masking_indices = None
            masking_indices_length = 0
            args.embed_dim = 0
        args.masking_indices = masking_indices

        # create policy model
        actor_backbone = MLP(input_dim=args.embed_dim + np.prod(args.obs_shape) - masking_indices_length, hidden_dims=args.actor_hidden_dims, mlp_initialization=True)
        data_actor_backbone = MLP(input_dim=args.embed_dim + np.prod(args.obs_shape) - masking_indices_length, hidden_dims=args.actor_hidden_dims, mlp_initialization=True)
        v_network_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, mlp_initialization=True)
        e_network_backbone = MLP(input_dim=args.embed_dim + np.prod(args.obs_shape) + args.action_dim - masking_indices_length, hidden_dims=args.hidden_dims, mlp_initialization=True)

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
        policy = OPTMAMLPolicy(
            actor=actor,
            data_actor=data_actor,
            v_network=v_network,
            e_network=e_network,
            encoder=encoder,
            v_network_optim=v_network_optim,
            e_network_optim=e_network_optim,
            encoder_optim=encoder_optim,
            masking_indices=masking_indices,
            args=args)

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
        policy_trainer = MFMAMLPolicyTrainer(
            policy=policy,
            eval_env=eval_env,
            training_buffers=training_buffers,
            testing_buffer=testing_buffer,
            logger=logger,
            epoch=args.epoch,
            rendering=args.rendering,
            step_per_epoch=args.step_per_epoch,
            batch_size=args.batch_size,
            num_trj=args.num_trj,
            eval_episodes=args.eval_episodes,
        )

        # train
        policy_trainer.maml_train(seed, normalize=args.normalize_obs)

if __name__ == "__main__":
    train()