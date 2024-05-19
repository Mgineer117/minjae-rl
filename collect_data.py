import argparse
import re
import random
import uuid
import os

import h5py
import torch
import gym
import numpy as np
import copy
import time

from datetime import date
today = date.today()

import metaworld
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.policy import SACPolicy

itr_re = re.compile(r'itr_(?P<itr>[0-9]+).pkl')

def find_env(args):
    if args.task_name is not None:
        env_key = args.env_type + '1' + '-' + args.task_name
        
        ml = ENVS[env_key]
        tasks = random.sample(ml.train_tasks, args.task_num)
        training_envs = []
        for task in tasks:
            env = ml.train_classes[args.task_name]()  # Create an environment with task `pick_place`
            env.set_task(task)
            training_envs.append(env)
        
        if args.env_type == 'ML':
            testing_envs = []
            tasks = random.choice(ml.test_tasks)
            for task in tasks:
                testing_envs.append(env.set_task(task))
        elif args.env_type == 'MT':
            testing_envs = copy.deepcopy(training_envs)

    else:
        env_key = args.env_type + str(args.task_num)
        ml = ENVS[env_key]
        
        training_envs = []
        for name, env_cls in ml.train_classes.items():
            env = env_cls()
            task = random.choice([task for task in ml.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            training_envs.append(env)

        if args.env_type == 'ML':
            testing_envs = []
            for name, env_cls in ml.test_classes.items():
                env = env_cls()
                task = random.choice([task for task in ml.test_tasks
                                        if task.env_name == name])
                env.set_task(task)
                testing_envs.append(env)
        elif args.env_type == 'MT':
            testing_envs = copy.deepcopy(training_envs)

    return training_envs, testing_envs

def load(pklfile):
    params = torch.load(pklfile)
    return params['trainer/policy']

def get_pkl_itr(pklfile):
    match = itr_re.search(pklfile)
    if match:
        return match.group('itr')
    raise ValueError(pklfile+" has no iteration number.")

def get_policy_wts(params):
    out_dict = {
        'fc0/weight': params.fcs[0].weight.data.numpy(),
        'fc0/bias': params.fcs[0].bias.data.numpy(),
        'fc1/weight': params.fcs[1].weight.data.numpy(),
        'fc1/bias': params.fcs[1].bias.data.numpy(),
        'last_fc/weight': params.last_fc.weight.data.numpy(),
        'last_fc/bias': params.last_fc.bias.data.numpy(),
        'last_fc_log_std/weight': params.last_fc_log_std.weight.data.numpy(),
        'last_fc_log_std/bias': params.last_fc_log_std.bias.data.numpy(),
    }
    return out_dict

def get_reset_data():
    data = dict(
        observations = [],
        next_observations = [],
        actions = [],
        rewards = [],
        terminals = [],
        timeouts = [],
        logprobs = [],
        successes = []
    )
    return data

def rollout(policy, training_envs, max_path, num_data, is_random=False, policy_training=False):
    unique_id = str(uuid.uuid4())[:5]
    folder_path = "model/" + unique_id
    print(f'folder path: {folder_path}')
    
    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    data = get_reset_data()
    current_step = 0
    while current_step < num_data:
        epoch = 0
        task_data = get_reset_data()
        for env in training_envs:
            traj_data = get_reset_data()
            # initialization
            _returns = 0
            t = 0 
            s, _ = env.reset()

            while t < max_path:
                if is_random:
                    a = env.action_space.sample()
                    logprob = np.log(1.0 / np.prod(env.action_space.high - env.action_space.low))
                else:
                    with torch.no_grad():
                        a, logprob = policy.actforward(s, deterministic=False)
                    a = a.cpu().numpy(); logprob = logprob.cpu().numpy()

                ns, rew, term, trunc, infos = env.step(a)
                
                _returns += rew
                t += 1

                if t == max_path:
                    trunc = True
                done = trunc or term

                traj_data['observations'].append(s)
                traj_data['actions'].append(a)
                traj_data['next_observations'].append(ns)
                traj_data['rewards'].append(rew)
                traj_data['terminals'].append(int(term))
                traj_data['timeouts'].append(int(trunc))
                traj_data['logprobs'].append(logprob)
                traj_data['successes'].append(infos['success'])

                s = ns
                if done:         
                    # clear log
                    epoch += 1
                    current_step += t
                    _returns = 0
                    s, _ = env.reset()
        
                    # merge data
                    for k in data:
                        task_data[k].extend(traj_data[k])
                    traj_data = get_reset_data()

        
        if policy_training:
            observations = np.array(task_data["observations"], dtype=np.float32)
            next_observations = np.array(task_data["next_observations"], dtype=np.float32)
            actions = np.array(task_data["actions"], dtype=np.float32)
            rewards = np.array(task_data["rewards"], dtype=np.float32).reshape(-1, 1)
            terminals = np.array(task_data["terminals"], dtype=np.int32).reshape(-1, 1)
            
            batch = {
                "observations": torch.tensor(observations).to(args.device),
                "actions": torch.tensor(actions).to(args.device),
                "next_observations": torch.tensor(next_observations).to(args.device),
                "terminals": torch.tensor(terminals).to(args.device),
                "rewards": torch.tensor(rewards).to(args.device)
            }
            loss_result = policy.learn(batch)
            torch.save(policy.actor.state_dict(), folder_path + '/model'+ '_ep(' + str(epoch) + ').pt')
        print('Len=%d, Returns=%f, Succ.Rate=%f, Progress:%d/%d' % (len(task_data["successes"]), np.sum(task_data["rewards"])/len(training_envs), np.sum(task_data["successes"])/len(training_envs), current_step, num_data))
        # merge task data
        for k in data:
            data[k].extend(task_data[k])
        task_data = get_reset_data()

    new_data = dict(
        observations=np.array(data['observations']).astype(np.float32),
        actions=np.array(data['actions']).astype(np.float32),
        next_observations=np.array(data['next_observations']).astype(np.float32),
        rewards=np.array(data['rewards']).astype(np.float32),
        terminals=np.array(data['terminals']).astype(np.int32),
        timeouts=np.array(data['timeouts']).astype(np.int32),
        successes=np.array(data['successes']).astype(np.int32)
    )
    new_data['infos/action_log_probs'] = np.array(data['logprobs']).astype(np.float32)

    for k in new_data:
        new_data[k] = new_data[k][:num_data]

    return new_data

ENVS = {
    #["ML1", "MT1", "ML10", "MT10", "ML45", "MT50"]
    'ML1-pick-place-v2': metaworld.ML1('pick-place-v2'),
    'MT1-pick-place-v2': metaworld.MT1('pick-place-v2'),
    #'ML1-push-v2': metaworld.ML1('push-v2'),
    #'MT1-push-v2': metaworld.MT1('push-v2'),

    #'ML1-reach-v2': metaworld.ML1('reach-v2'),
    #'MT1-reach-v2': metaworld.MT1('reach-v2'),
    #'ML1-sweep-into-v2': metaworld.ML1('sweep-into-v2'),
    #'MT1-sweep-into-v2': metaworld.MT1('sweep-into-v2'),
    #'ML1-window-open-v2': metaworld.ML1('window-open-v2'),
    #'MT1-window-open-v2': metaworld.MT1('window-open-v2'),

    #'ML1-basketball-v2': metaworld.ML1('basketball-v2'),
    #'MT1-basketball-v2': metaworld.MT1('basketball-v2'),
    #'ML1-button-press-v2': metaworld.ML1('button-press-v2'),
    #'MT1-button-press-v2': metaworld.MT1('button-press-v2'),
    #'ML1-dial-turn-v2': metaworld.ML1('dial-turn-v2'),
    #'MT1-dial-turn-v2': metaworld.MT1('dial-turn-v2'),
    #'ML1-drawer-close-v2': metaworld.ML1('drawer-close-v2'),
    #'MT1-drawer-close-v2': metaworld.MT1('drawer-close-v2'),
    #'ML1-peg-insert-side-v2': metaworld.ML1('peg-insert-side-v2'),
    #'MT1-peg-insert-side-v2': metaworld.MT1('peg-insert-side-v2'),
    #####################################################################
    #'ML10': metaworld.ML10(),
    #'MT10': metaworld.MT10(),
    #'ML45': metaworld.ML45(),
    #'MT50': metaworld.MT50(),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', type=str, default='MT') # ML or MT
    parser.add_argument('--task-name', type=str, default="pick-place-v2") # MT1 or MT 10, 45, 50
    parser.add_argument('--task-num', type=int, default=1) # 10, 45, 50
    parser.add_argument('--pklfile', type=str, default=None)
    parser.add_argument('--actor-hidden-dims', default=(256, 256))
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--max_path', type=int, default=500)
    parser.add_argument('--num_data', type=int, default=1000000)
    parser.add_argument('--is-random', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### Env generation
    training_envs, _ = find_env(args)
    
    ### Policy
    policy = None
    if not args.is_random:
        args.obs_shape = (training_envs[0].observation_space.shape[0],)
        args.action_dim = np.prod(training_envs[0].action_space.shape)
        args.max_action = training_envs[0].action_space.high[0]
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
        if args.pklfile is None:
            policy_training = True
        else:
            policy.load_state_dict(torch.load('model/model.pt'))
            policy_training = False

    data = rollout(policy, training_envs, max_path=args.max_path, num_data=args.num_data, is_random=args.is_random, policy_training=policy_training)
    rightnow = today.strftime("%m_%d_%y")    
    
    if args.task_name is not None:
        words = [rightnow, args.env_type + '1', args.task_name + '.hdf5']
        hfile_name = "-".join(words)
    else:
        words = [rightnow, args.env_type + str(args.task_num), args.task_name + '.hdf5']
        hfile_name = "-".join(words)
    
    hile_path = os.path.join('generated_data', hfile_name) 
    hfile = h5py.File(hile_path, 'w')
    for k in data:
        hfile.create_dataset(k, data=data[k], compression='gzip')

    '''
    if args.is_random:
        pass
    else:
        hfile['metadata/algorithm'] = np.string_('SAC')
        hfile['metadata/iteration'] = np.array([get_pkl_itr(args.pklfile)], dtype=np.int32)[0]
        hfile['metadata/policy/nonlinearity'] = np.string_('relu')
        hfile['metadata/policy/output_distribution'] = np.string_('tanh_gaussian')
        for k, v in get_policy_wts(policy).items():
            hfile['metadata/policy/'+k] = v
    '''
    hfile.close()
