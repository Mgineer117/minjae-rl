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
from rlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from rlkit.dynamics import EnsembleDynamics
from rlkit.utils.scaler import StandardScaler
from rlkit.utils.termination_fns import get_termination_fn
from rlkit.utils.load_dataset import qlearning_dataset
from rlkit.buffer import ReplayBuffer
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy_trainer import MBPolicyTrainer
from rlkit.policy import MOBILEPolicy


"""
suggested hypers
halfcheetah-random-v2: rollout-length=5, penalty-coef=0.5
hopper-random-v2: rollout-length=5, penalty-coef=5.0
walker2d-random-v2: rollout-length=5, penalty-coef=2.0
halfcheetah-medium-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-v2: rollout-length=5, penalty-coef=1.5 auto-alpha=False
walker2d-medium-v2: rollout-length=5, penalty-coef=0.5
halfcheetah-medium-replay-v2: rollout-length=5, penalty-coef=0.1
hopper-medium-replay-v2: rollout-length=5, penalty-coef=0.1
walker2d-medium-replay-v2: rollout-length=1, penalty-coef=0.5
halfcheetah-medium-expert-v2: rollout-length=5, penalty-coef=2.0
hopper-medium-expert-v2: rollout-length=5, penalty-coef=1.5
walker2d-medium-expert-v2: rollout-length=1, penalty-coef=1.5
"""

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="popodice")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--algo-name", type=str, default="mobile")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument('--seeds', default=[1, 3, 5, 7, 9], type=list)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument('--actor-hidden-dims', default=(64, 64))
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=bool, default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--num-q-ensemble", type=int, default=2)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--norm-reward", type=bool, default=False)

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs-since-update", type=int, default=5)
    parser.add_argument("--dynamics-max-epochs", type=int, default=30)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--penalty-coef", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr-scheduler", type=bool, default=True)
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
        if args.norm_reward:
            r_mean, r_std = dataset["rewards"].mean(), dataset["rewards"].std()
            dataset["rewards"] = (dataset["rewards"] - r_mean) / (r_std + 1e-3)

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
        dist = TanhDiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=args.action_dim,
            unbounded=True,
            conditioned_sigma=True,
            max_mu=args.max_action
        )
        actor = ActorProb(actor_backbone, dist, args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critics = []
        for i in range(args.num_q_ensemble):
            critic_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
            critics.append(Critic(critic_backbone, args.device))
        critics = torch.nn.ModuleList(critics)
        critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)

        if args.lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
        else:
            lr_scheduler = None

        if args.auto_alpha:
            target_entropy = args.target_entropy if args.target_entropy \
                else -np.prod(env.action_space.shape)

            args.target_entropy = target_entropy

            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)
        else:
            alpha = args.alpha

        # create dynamics
        load_dynamics_model = True if args.load_dynamics_path else False
        dynamics_model = EnsembleDynamicsModel(
            obs_dim=np.prod(args.obs_shape),
            action_dim=args.action_dim,
            hidden_dims=args.dynamics_hidden_dims,
            num_ensemble=args.n_ensemble,
            num_elites=args.n_elites,
            weight_decays=args.dynamics_weight_decay,
            device=args.device
        )
        dynamics_optim = torch.optim.Adam(
            dynamics_model.parameters(),
            lr=args.dynamics_lr
        )
        scaler = StandardScaler()
        termination_fn = get_termination_fn(task=args.task)
        dynamics = EnsembleDynamics(
            dynamics_model,
            dynamics_optim,
            scaler,
            termination_fn
        )

        if args.load_dynamics_path:
            dynamics.load(args.load_dynamics_path)

        # create policy
        policy = MOBILEPolicy(
            dynamics,
            actor,
            critics,
            actor_optim,
            critics_optim,
            tau=args.tau,
            gamma=args.gamma,
            alpha=alpha,
            penalty_coef=args.penalty_coef,
            num_samples=args.num_samples,
            deterministic_backup=args.deterministic_backup,
            max_q_backup=args.max_q_backup
        )

        domain_name = args.task.split('-')[0]
        pomdp = MASKING_IDX[domain_name] if args.pomdp == True else None

        # create buffer
        real_buffer = ReplayBuffer(
            buffer_size=len(dataset["observations"]),
            obs_shape=args.obs_shape,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32,
            pomdp=pomdp,
            device=args.device
        )
        real_buffer.load_dataset(dataset)

        fake_buffer = ReplayBuffer(
            buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
            obs_shape=args.obs_shape,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32,
            pomdp=pomdp,
            device=args.device
        )

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
        policy_trainer = MBPolicyTrainer(
            policy=policy,
            eval_env=env,
            real_buffer=real_buffer,
            fake_buffer=fake_buffer,
            logger=logger,
            rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
            epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            batch_size=args.batch_size,
            real_ratio=args.real_ratio,
            eval_episodes=args.eval_episodes,
            lr_scheduler=lr_scheduler,
            pomdp=pomdp
        )

        # train
        if not load_dynamics_model:
            dynamics.train(
                real_buffer.sample_all(),
                logger,
                max_epochs_since_update=args.max_epochs_since_update,
                max_epochs=args.dynamics_max_epochs
            )
        
        policy_trainer.train(seed)


if __name__ == "__main__":
    testing_param = [[1, 3, 5, 7, 9]]
    for param in testing_param:
        train(param=param)