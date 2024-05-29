import sys
sys.path.append('../minjae-rl')

import uuid
import argparse
import random
import os

import numpy as np
import torch

from rlkit.utils.utils import seed_all, select_device
from rlkit.nets import MLP, OneHotEncoder, RecurrentEncoder
from rlkit.modules import ActorProb, Critic, DistCritic, PhiNetwork, DiagGaussian
from rlkit.utils.load_dataset import qlearning_dataset
from rlkit.utils.load_env import load_env
from rlkit.utils.load_cost_fn import load_cost_fn
from rlkit.utils.zfilter import ZFilter
from rlkit.buffer import OnlineSampler
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy_trainer import MFPolicyTrainer
from rlkit.policy import PPOPolicy

def get_args():
    parser = argparse.ArgumentParser()
    '''WandB and Logging parameters'''
    parser.add_argument("--project", type=str, default="purpose")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument('--task', type=str, default=None) # None for Gym and MetaGym except ML1 or MT1
    parser.add_argument("--algo-name", type=str, default="mtppo")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="log")

    '''OpenAI Gym parameters'''
    parser.add_argument('--env-type', type=str, default='MetaGym') # Gym or MetaGym
    parser.add_argument('--agent-type', type=str, default='MT10') # MT1, ML45, Hopper, Ant
    parser.add_argument('--task-name', type=str, default=None) # None for Gym and MetaGym except ML1 or MT1 'pick-place'
    parser.add_argument('--task-num', type=int, default=None) # 10, 45, 50

    '''Algorithmic and sampling parameters'''
    parser.add_argument("--embed-type", type=str, default='purpose') # onehot or purpose
    parser.add_argument("--embed-dim", type=int, default=5) # one-hot or purpose
    parser.add_argument('--seeds', default=[1, 3, 5, 7, 9], type=list)
    parser.add_argument('--num-cores', type=int, default=None)
    parser.add_argument('--actor-hidden-dims', default=(256, 256))
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument("--K-epochs", type=int, default=5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=50)
    parser.add_argument('--episode_len', type=int, default=500)
    parser.add_argument('--episode_num', type=int, default=2)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--rendering", type=bool, default=True)
    parser.add_argument("--import-policy", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()

def train(args=get_args()):
    unique_id = str(uuid.uuid4())[:4]
    args.device = select_device()
    
    for seed in args.seeds:
        # seed
        seed_all(seed)

        # create env and dataset
        args.task = '-'.join((args.env_type, args.agent_type))
        training_envs, testing_envs, eval_env_idx = load_env(args.task, args.task_name, args.task_num)
        cost_fn = load_cost_fn(args.task)

        # create policy model
        '''state dimension input manipulation for Ss only'''
        args.obs_shape = (training_envs[0].observation_space.shape[0],)
        args.action_dim = np.prod(training_envs[0].action_space.shape)
        args.max_action = training_envs[0].action_space.high[0]
        if args.embed_type == 'onehot': # for multi-task only
            args.embed_dim = len(training_envs)
            encoder = OneHotEncoder(
                embed_dim=args.embed_dim,
                eval_env_idx=eval_env_idx,
                device = args.device
            )
            encoder_optim = None
        elif args.embed_type == 'purpose':
            rnn_size = int(np.prod(args.obs_shape) + args.action_dim + np.prod(args.obs_shape) + 1)
            encoder = RecurrentEncoder(
                input_size=rnn_size, 
                hidden_size=rnn_size, 
                output_size=args.embed_dim,
                device = args.device
            )
            encoder_optim = torch.optim.Adam(encoder.parameters(), lr=args.critic_lr)
        else:
            print('...No task embedding is applied')
            encoder_optim = None
            args.embed_dim = 0

        actor_backbone = MLP(input_dim=args.embed_dim + np.prod(args.obs_shape), hidden_dims=args.actor_hidden_dims, activation=torch.nn.Tanh)
        critic_backbone = MLP(input_dim=args.embed_dim + np.prod(args.obs_shape), hidden_dims=args.hidden_dims, activation=torch.nn.Tanh,)
        
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
                
        critic = Critic(critic_backbone, device=args.device)

        optimizer = torch.optim.AdamW([
                        {'params': actor.parameters(), 'lr': args.actor_lr},
                        {'params': critic.parameters(), 'lr': args.critic_lr}
                    ])

        # Define sampler (online) or buffer (offline)
        running_state = ZFilter(args.obs_shape, clip=5)

        sampler = OnlineSampler(
            obs_shape=args.obs_shape,
            action_dim=args.action_dim,
            episode_len=args.episode_len,
            episode_num=args.episode_num,
            training_envs=training_envs,
            cost_fn=cost_fn,
            running_state=running_state,
            num_cores=args.num_cores,
            device=args.device,
        )
        
        # define combinatory policy
        policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            encoder=encoder,
            encoder_optim=encoder_optim,
            K_epochs=args.K_epochs,
            eps_clip=args.eps_clip,
            device=args.device
        )

        # setup logger
        default_cfg = vars(args)#asdict(args)
        if args.name is None:
            args.name = args.algo_name + '-' + args.task + '-' + unique_id + "-seed" + str(seed)
        if args.group is None:
            args.group = args.task + "-seed-" + str(seed)
        if args.logdir is not None:
            args.logdir = os.path.join(args.logdir, args.name, args.group)  
        logger = WandbLogger(default_cfg, args.project, args.group, args.name, args.logdir)
        logger.save_config(default_cfg, verbose=args.verbose)

        # create policy trainer
        policy_trainer = MFPolicyTrainer(
            policy=policy,
            eval_env=testing_envs,
            eval_env_idx=eval_env_idx,
            cost_fn=cost_fn,
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
        policy_trainer.online_train(seed)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()