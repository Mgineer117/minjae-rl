import sys
sys.path.append('../minjae-rl')

import uuid
import argparse
import gym
import random
import os
import pickle

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rlkit.utils.utils import seed_all, select_device, call_encoder
from rlkit.nets import MLP, RNNModel, RecurrentEncoder, BaseEncoder, OneHotEncoder
from rlkit.modules import ActorProb, Critic, DistCritic, DiagGaussian
from rlkit.utils.load_dataset import qlearning_dataset
from rlkit.utils.load_env import load_metagym_env, load_gym_env
from rlkit.utils.load_reward_fn import load_reward_fn
from rlkit.utils.load_cost_fn import load_cost_fn
from rlkit.utils.zfilter import ZFilter
from rlkit.buffer import OnlineSampler
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy_trainer import MFPolicyTrainer
from rlkit.policy import PPOPolicy

def get_args():
    parser = argparse.ArgumentParser()
    '''WandB and Logging parameters'''
    parser.add_argument("--project", type=str, default="OMRL")
    parser.add_argument('--task', type=str, default=None) # None for Gym and MetaGym except ML1 or MT1
    parser.add_argument("--algo-name", type=str, default="ppo")
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument('--log-interval', type=int, default=10)

    '''OpenAI Gym parameters'''
    parser.add_argument('--env-type', type=str, default='MetaGym') # Gym or MetaGym
    parser.add_argument('--agent-type', type=str, default='ML10') # MT1, ML45, Hopper, Ant
    parser.add_argument('--task-name', type=str, default=None) # None for Gym and MetaGym except ML1 or MT1 'pick-place'
    parser.add_argument('--task-num', type=int, default=None) # 10, 45, 50

    '''Algorithmic and sampling parameters'''
    parser.add_argument('--seeds', default=[1, 3, 5, 7, 9], type=list)
    parser.add_argument('--num-cores', type=int, default=None)
    parser.add_argument('--actor-hidden-dims', default=(256, 256))
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument("--K-epochs", type=int, default=5)
    parser.add_argument("--eps-clip", type=float, default=0.1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=7e-4)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--embed-type", type=str, default='skill') # skill, task, onehot, or none
    parser.add_argument("--embed-loss", type=str, default='decoder') # action or reward or decoder
    parser.add_argument("--embed-dim", type=int, default=5)
    parser.add_argument("--policy-mask-type", type=str, default='ego') # ego or other or none # this is for skill embedding
    parser.add_argument("--decoder-mask-type", type=str, default='ego') # ego or other or none # this is for skill embedding

    '''Sampling parameters'''
    parser.add_argument('--epoch', type=int, default=20000)
    parser.add_argument('--init-epoch', type=int, default=0)
    parser.add_argument("--step-per-epoch", type=int, default=50)
    parser.add_argument('--episode_len', type=int, default=1000)
    parser.add_argument('--episode_num', type=int, default=2)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--rendering", type=bool, default=True)
    parser.add_argument("--visualize-latent-space", type=bool, default=True)
    parser.add_argument("--data_num", type=int, default=None)
    parser.add_argument("--import-policy", type=bool, default=False)
    parser.add_argument("--gpu-idx", type=int, default=0)
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()

def train(args=get_args()):
    unique_id = str(uuid.uuid4())[:4]
    args.device = select_device(args.gpu_idx)

    for seed in args.seeds:
        # seed
        seed_all(seed)

        # create env and dataset
        args.task = '-'.join((args.env_type, args.agent_type))
        if args.env_type =='Gym':
            # For Gym, we use multiple reward dynamics for multiple tasks
            if args.task_num >= 2:
                reward_fn_list = load_reward_fn(args.task, num_task=args.task_num)
            else:
                reward_fn_list = None
            training_envs, testing_envs, eval_env_idx = load_gym_env(args.task, reward_fn=reward_fn_list)
        elif args.env_type =='MetaGym':
            training_envs, testing_envs, eval_env_idx = load_metagym_env(args.task, args.task_name, args.task_num, render_mode='rgb_array')
        else:
            NotImplementedError

        # get dimensional parameters
        args.obs_shape = (training_envs[0].observation_space.shape[0],)
        args.action_dim = np.prod(training_envs[0].action_space.shape)
        args.max_action = training_envs[0].action_space.high[0]
        optim_params = []

        # define encoder 
        encoder, optim_params = call_encoder(training_envs, eval_env_idx, optim_params, args)

        # define necessary ingredients for training
        #running_state = ZFilter(args.obs_shape, clip=5)
        # do not use running state for encoding method
        running_state = None

        actor_backbone = MLP(input_dim=args.embed_dim + np.prod(args.obs_shape) - len(args.masking_indices), hidden_dims=args.actor_hidden_dims, activation=torch.nn.Tanh,)
        critic_backbone = MLP(input_dim=args.embed_dim + np.prod(args.obs_shape) - len(args.masking_indices), hidden_dims=args.hidden_dims, activation=torch.nn.Tanh,)
        
        dist = DiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=args.action_dim,
            unbounded=False,
            conditioned_sigma=True,
            max_mu=args.max_action,
            sigma_min=-3.0,
            sigma_max=2.0
        )

        actor = ActorProb(actor_backbone,
                          dist_net=dist,
                          device=args.device)   
                
        critic = Critic(critic_backbone, device = args.device)

        optim_params.append({'params': actor.parameters(), 'lr': args.actor_lr})
        optim_params.append({'params': critic.parameters(), 'lr': args.critic_lr})
        optimizer = torch.optim.Adam(optim_params)
        
        # import pre-trained model before defining actual models
        if args.import_policy:
            try:
                actor, critic, encoder, running_state = pickle.load(open('model/model.p', "rb"))
            except:
                actor, critic, encoder = pickle.load(open('model/model.p', "rb"))
        
        # define training components
        sampler = OnlineSampler(
            obs_shape=args.obs_shape,
            action_dim=args.action_dim,
            embed_dim=args.embed_dim,
            episode_len=args.episode_len,
            episode_num=args.episode_num,
            training_envs=training_envs,
            running_state=running_state,
            num_cores=args.num_cores,
            data_num=args.data_num,
            device=args.device,
        )

        policy = PPOPolicy(
            actor=actor,
            critic=critic,
            encoder=encoder,
            optimizer=optimizer,
            masking_indices=args.masking_indices,
            decoder_masking_indices=args.decoder_masking_indices,
            embed_loss=args.embed_loss,
            K_epochs=args.K_epochs,
            eps_clip=args.eps_clip,
            device=args.device
        )

        # setup logger
        default_cfg = vars(args)#asdict(args)
        args.group = '-'.join((args.task, args.algo_name, unique_id))
        args.name = '-'.join((args.algo_name, unique_id, "seed:" + str(seed)))
        args.logdir = os.path.join(args.logdir, args.group)  
            
        logger = WandbLogger(default_cfg, args.project, args.group, args.name, args.logdir)
        logger.save_config(default_cfg, verbose=args.verbose)

        tensorboard_path = os.path.join(logger.log_dir, 'tensorboard')
        os.mkdir(tensorboard_path)
        writer = SummaryWriter(log_dir=tensorboard_path)

        # create policy trainer
        policy_trainer = MFPolicyTrainer(
            policy=policy,
            eval_env=testing_envs,
            eval_env_idx=eval_env_idx,
            sampler=sampler,
            logger=logger,
            writer=writer,
            epoch=args.epoch,
            init_epoch=args.init_epoch,
            step_per_epoch=args.step_per_epoch,
            eval_episodes=args.eval_episodes,
            rendering=args.rendering,
            obs_dim=args.obs_shape[0],
            action_dim=args.action_dim,
            embed_dim=args.embed_dim,
            log_interval=args.log_interval,
            visualize_latent_space=args.visualize_latent_space,
            device=args.device,
        )

        # train
        policy_trainer.online_train(seed)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()