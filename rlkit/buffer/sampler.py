import random
import time
import math
import h5py
import os
import torch.multiprocessing as multiprocessing

import torch
import numpy as np

from typing import Optional, Union, Tuple, Dict
from datetime import date
today = date.today()

def cost_fn(s, a, ns):
    cost = 0.0
    return cost

def calculate_workers_and_rounds(environments, episodes_per_env, num_cores):
    if episodes_per_env == 1:
        num_worker_per_env = 1
    elif episodes_per_env >= 2:
        num_worker_per_env = episodes_per_env // 2
    
    # Calculate total number of workers
    total_num_workers = num_worker_per_env * len(environments)

    if total_num_workers > num_cores:
        rounds = math.ceil(total_num_workers / num_cores) 

        num_worker_per_round = []
        workers_remaining = total_num_workers
        for i in range(rounds):
            if workers_remaining >= num_cores:
                num_worker_per_round.append(num_cores)
                workers_remaining -= num_cores
            else:
                num_worker_per_round.append(workers_remaining)
                workers_remaining = 0
        num_env_per_round = [int(x / num_worker_per_env) for x in num_worker_per_round] #num_worker_per_round / num_worker_per_env
    else:
        rounds = 1
        num_worker_per_round = [total_num_workers]
        num_env_per_round = [len(environments)]
    
    episodes_per_worker = int(episodes_per_env * len(environments) / total_num_workers)
    return num_worker_per_round, num_env_per_round, episodes_per_worker, rounds

class OnlineSampler:
    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        episode_len: int,
        episode_num: int,
        training_envs: list,
        running_state = None,
        data_num: int = None,
        device: str = "cpu"
    ) -> None:
        self.obs_dim = obs_shape[0]
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.episode_num = episode_num
        self.training_envs = training_envs
        self.running_state = running_state
        self.data_num = data_num

        self.device = torch.device(device)

        # Preprocess for multiprocessing to avoid CPU overscription and deadlock
        self.num_cores = multiprocessing.cpu_count() #torch.get_num_threads()
        num_workers_per_round, num_env_per_round, episodes_per_worker, rounds = calculate_workers_and_rounds(self.training_envs, self.episode_num, self.num_cores)
        
        self.num_workers_per_round = num_workers_per_round
        self.num_env_per_round = num_env_per_round
        self.episodes_per_worker = episodes_per_worker
        self.thread_batch_size = self.episodes_per_worker * self.episode_len
        self.rounds = rounds

        print('Sampling Parameters:')
        print('--------------------')
        print(f'Core usage for this run           : {self.num_workers_per_round[0]}/{self.num_cores}')
        print(f'Number of Environments each Round : {self.num_env_per_round}')
        print(f'Episodes per Worker               : {self.episodes_per_worker}')
        torch.set_num_threads(1) # enforce one task for each worker to avoide CPU overscription.

        if self.data_num is not None:
            # to create an enough batch..
            self.data_buffer = self.get_reset_data(self.data_num+self.episode_len)
            self.buffer_last_idx = 0

    def save_buffer(self):
        for k in self.data_buffer:
            self.data_buffer[k] = self.data_buffer[k][:self.data_num]
        print('mean reward: ',np.mean(self.data_buffer['rewards']))
        print('mean cost: ',np.mean(self.data_buffer['costs']))
        hfile = h5py.File('data.h5py', 'w')
        for k in self.data_buffer:
            hfile.create_dataset(k, data=self.data_buffer[k], compression='gzip')
        hfile.close()

    def get_reset_data(self, batch_size):
        '''
        We create a initialization batch to avoid the daedlocking. 
        The remainder of zero arrays will be cut in the end.
        '''
        data = dict(
            observations = np.zeros((batch_size, self.obs_dim)),
            next_observations = np.zeros((batch_size, self.obs_dim)),
            actions = np.zeros((batch_size, self.action_dim)),
            rewards = np.zeros((batch_size, 1)),
            costs = np.zeros((batch_size, 1)),
            terminals = np.zeros((batch_size, 1)),
            timeouts = np.zeros((batch_size, 1)),
            masks = np.zeros((batch_size, 1)),
            logprobs = np.zeros((batch_size, 1)),
            env_idxs = np.zeros((batch_size, 1)),
            successes = np.zeros((batch_size, 1)),
        )
        return data

    def collect_trajectory(self, pid, queue, env, policy, thread_batch_size, episode_len,
                           deterministic=False, env_idx=0, seed=0):
        # estimate the batch size to hava a large batch
        batch_size = thread_batch_size + episode_len
        data = self.get_reset_data(batch_size=batch_size)

        current_step = 0
        while current_step < thread_batch_size:
            # var initialization
            _returns = 0
            t = 0 

            # env initialization
            try:
                s, _ = env.reset(seed=seed)
            except:
                s = env.reset(seed=seed)
            
            # normalizing state
            if self.running_state is not None:
                s = self.running_state(s)
            # create mdp for encoding process. all element should have dimension (1,) than scaler
            a = np.zeros((self.action_dim, ))
            ns = s

            mdp = (s, a, ns, np.array([0]), np.array([1]))

            # policy.encode should output s, ns, encoded_s, and encodded_ns
            with torch.no_grad():
                s, _, e_s, _ = policy.encode_obs(mdp, env_idx=env_idx)
            
            # begin the episodic loop
            while t < episode_len:
                # sample action
                with torch.no_grad():
                    a, logprob = policy.select_action(e_s, deterministic=deterministic)
                # env stepping
                try:
                    ns, rew, term, trunc, infos = env.step(a)
                except:
                    ns, rew, term, infos = env.step(a)                    
                    trunc = True if t == episode_len else False
                try:
                    success = infos['success']
                except:
                    success = 0.0 
                cost = cost_fn(s, a, ns)
                done = trunc or term
                mask = 0 if done else 1
                
                # normalizing state
                if self.running_state is not None:
                    ns = self.running_state(ns)

                # state encoding
                mdp = (s, a, ns, np.array([rew]), np.array([mask]))
                with torch.no_grad():
                    _, ns, _, e_ns = policy.encode_obs(mdp, env_idx=env_idx, reset=False)

                s = ns; e_s = e_ns
                
                # saving the data
                data['observations'][current_step+t, :] = s
                data['actions'][current_step+t, :] = a
                data['next_observations'][current_step+t, :] = ns
                data['rewards'][current_step+t, :] = rew
                data['costs'][current_step+t, :] = cost
                data['terminals'][current_step+t, :] = term
                data['timeouts'][current_step+t, :] = trunc
                data['masks'][current_step+t, :] = mask
                data['logprobs'][current_step+t, :] = logprob
                data['env_idxs'][current_step+t, :] = env_idx    
                data['successes'][current_step+t, :] = success
            
                _returns += rew
                t += 1
    
                if done:        
                    # clear log
                    current_step += t
                    _returns = 0
                    try:
                        s, _ = env.reset(seed=seed)
                    except:
                        s = env.reset(seed=seed)
                    break
        
        memory = dict(
            observations=data['observations'].astype(np.float32),
            actions=data['actions'].astype(np.float32),
            next_observations=data['next_observations'].astype(np.float32),
            rewards=data['rewards'].astype(np.float32),
            costs=data['costs'].astype(np.float32),
            terminals=data['terminals'].astype(np.int32),
            timeouts=data['timeouts'].astype(np.int32),
            masks=data['masks'].astype(np.int32),
            logprobs=data['logprobs'].astype(np.float32),
            env_idxs=data['env_idxs'].astype(np.int32),
            successes=data['successes'].astype(np.float32),
        )

        if self.data_num is not None:
            for k in memory:
                self.data_buffer[k][self.buffer_last_idx:self.buffer_last_idx+current_step, :] = memory[k] 
            self.buffer_last_idx += current_step
            if self.buffer_last_idx >= self.data_num:
                self.save_buffer()
                self.data_num = None

        for k in memory:
            memory[k] = memory[k][:thread_batch_size]
                                
        if queue is not None:
            queue.put([pid, memory])
        else:
            return memory

    def to_device(self, policy, device=torch.device('cpu')):
        policy = policy.to(device)
        policy.device = device
        policy.actor.device = device
        policy.encoder.device = device
        return policy

    def collect_samples(self, policy, seed, deterministic=False):
        '''
        All sampling and saving to the memory is done in numpy.
        return: dict() with elements in numpy
        '''
        t_start = time.time()
        policy = self.to_device(policy)
        
        queue = multiprocessing.Manager().Queue()
        env_idx = 0
        worker_idx = 0
        for round_number in range(self.rounds):
            #print(f"Starting round {round_number + 1}/{self.rounds}")
            processes = []
            
            #print(f'indices: {env_idx}<->{env_idx+self.num_env_per_round[round_number]}')
            envs = self.training_envs[env_idx:env_idx+self.num_env_per_round[round_number]]
            for env in envs:
                workers_for_env = self.num_workers_per_round[round_number] // len(envs)
                for _ in range(workers_for_env):
                    worker_args = (worker_idx, queue, env, policy, self.thread_batch_size, 
                               self.episode_len, deterministic, env_idx, seed)
                    p = multiprocessing.Process(target=self.collect_trajectory, args=worker_args)
                    processes.append(p)
                    p.start()
                    worker_idx += 1
                env_idx += 1
            
            for p in processes:
                p.join()
            
        worker_memories = [None] * worker_idx
        for _ in range(worker_idx): 
            pid, worker_memory = queue.get()
            worker_memories[pid] = worker_memory
        
        memory = worker_memories[0]
        for worker_memory in worker_memories[1:]:
            for k in memory:
                memory[k] = np.concatenate((memory[k], worker_memory[k]), axis=0)

        policy = self.to_device(policy, self.device)
        t_end = time.time()

        return memory, t_end - t_start
