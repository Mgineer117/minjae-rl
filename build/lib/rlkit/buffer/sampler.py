import random
import time
import math
import h5py
import os
import multiprocessing
from multiprocessing import set_start_method

import torch
import numpy as np

from typing import Optional, Union, Tuple, Dict
from datetime import date
today = date.today()

def cost_fn(s, a, ns):
    cost = 0
    if np.abs(ns[0]) > 0.3:
        cost += 1
    return cost

class OnlineSampler:
    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        episode_len: int,
        episode_num: int,
        training_envs: list,
        running_state = None,
        track_data: bool = False,
        pomdp: list = None,
        device: str = "cpu"
    ) -> None:
        self.obs_dim = obs_shape[0]
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.episode_num = episode_num
        self.training_envs = training_envs
        self.running_state = running_state
        self.track_data = track_data
        self.data_size = 1_000_000
        self.pomdp = pomdp

        self.device = torch.device(device)

        self.num_worker_per_env = 1 if (self.episode_num / 2) <= 1 else math.ceil(self.episode_num / 2)
        self.total_num_workers = int(len(self.training_envs) * self.num_worker_per_env)
        self.thread_batch_size = int(self.episode_num * self.episode_len / self.num_worker_per_env)
        self.queue = multiprocessing.Queue()

        if self.track_data:
            self.memory = dict(
            observations=[],
            actions=[],
            next_observations=[],
            rewards=[],
            costs=[],
            terminals=[],
            timeouts=[],
            masks=[],
            logprobs=[],
            env_idxs=[],
            successes=[],
            )

    def get_reset_data(self, batch_size):
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

    def make_pomdp(self):
        # not developed
        observations = np.delete(self.observations, self.pomdp, axis=1)
        next_observations = np.delete(self.next_observations, self.pomdp, axis=1)
        initial_observations = np.delete(self.initial_observations, self.pomdp, axis=1)
        if self._obs_normalized:
            self._obs_mean = np.delete(self._obs_mean, self.pomdp)
            self._obs_std = np.delete(self._obs_std, self.pomdp)

        self.observations = observations    
        self.next_observations = next_observations
        self.initial_observations = initial_observations

    def collect_trajectory(self, pid, queue, env, policy, thread_batch_size, episode_len,
                           deterministic=False, running_state=None, env_idx=0, seed=0):
        # estimate the batch size
        batch_size = thread_batch_size + episode_len
        data = self.get_reset_data(batch_size=batch_size)
        current_step = 0
        while current_step < thread_batch_size:
            # initialization
            _returns = 0
            t = 0 
            try:
                s, _ = env.reset(seed=seed)
            except:
                s = env.reset(seed=seed)
            a = np.zeros((self.action_dim, ))
            ns = s # initialization
            rew = 0.0
            mask = 1
            
            s, _, e_s, _ = policy.encode_obs((s, a, ns, [rew], mask), running_state=running_state, env_idx=env_idx)

            while t < episode_len:
                with torch.no_grad():
                    a, logprob = policy.actforward(e_s, deterministic=deterministic)
                    a = a.numpy(); logprob = logprob.numpy()

                try:
                    ns, rew, term, trunc, infos = env.step(a)
                except:
                    ns, rew, term, infos = env.step(a)                    
                    trunc = True if t == episode_len else False
                
                _, ns, _, e_ns = policy.encode_obs((s, a, ns, [rew], mask), running_state=running_state, env_idx=env_idx, reset=False)
                s = ns; e_s = e_ns
                
                cost = cost_fn(s, a, ns)

                done = trunc or term
                mask = 0 if done else 1

                _returns += rew

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
                try:
                    data['successes'][current_step+t, :] = infos['success']
                except:
                    data['successes'][current_step+t, :] = 0.0

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
            logprobs=data['logprobs'].astype(np.int32),
            env_idxs=data['env_idxs'].astype(np.int32),
            successes=data['successes'].astype(np.float32),
        )
        if self.track_data:
            for k in self.memory:
                self.memory[k].extend(memory[k])
            if len(self.memory['rewards']) >= self.data_size:
                for k in self.memory:
                    self.memory[k] = self.memory[k][:self.data_size]
                print('mean reward: ',np.mean(self.memory['rewards']))
                print('mean cost: ',np.mean(self.memory['costs']))
                hfile = h5py.File('data.h5py', 'w')
                for k in self.memory:
                    hfile.create_dataset(k, data=self.memory[k], compression='gzip')
                hfile.close()

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

    def collect_samples(self, training_envs, policy, seed, deterministic=False):
        '''
        It is designed for one worker to work on two episodes, and one worker at least is assigned for each env. 
        At least one worker for one env,
        One worker to take two episodes sampling.
        Hence, total num_workers = len(training_envs) * (episode_num / 2)
        So targetting worker_batch_size will be: total / num_worker 
            = (len(training_envs) * episode_len * episode_num) / (len(training_envs) * (episode_num / 2))
            = 2 * episode_len
        '''
        t_start = time.time()
        policy = self.to_device(policy)
        
        if self.total_num_workers != 1:
            workers = []
            for i, env in enumerate(self.training_envs):
                for j in range(self.num_worker_per_env):
                    worker_idx = i*self.num_worker_per_env + j + 1
                    if worker_idx == self.total_num_workers:
                        break
                    else:
                        worker_args = (worker_idx, self.queue, env, policy, self.thread_batch_size, self.episode_len,
                                        deterministic, self.running_state, i, seed)
                        workers.append(multiprocessing.Process(target=self.collect_trajectory, args=worker_args))
        
            for worker in workers:
                worker.start()

        memory = self.collect_trajectory(0, None, training_envs[-1], policy, self.thread_batch_size, self.episode_len,
                                        deterministic, self.running_state, len(training_envs)-1, seed)
        
        if self.total_num_workers != 1:
            worker_memories = [None] * len(workers)
            for worker in workers: 
                pid, worker_memory = self.queue.get()
                worker_memories[pid - 1] = worker_memory
            for worker_memory in worker_memories:
                for k in memory:
                    memory[k] = np.concatenate((memory[k], worker_memory[k]), axis=0)
                    
        policy = self.to_device(policy, self.device)
        t_end = time.time()
        
        for key, item in memory.items():
            memory[key] = torch.tensor(item).to(self.device)
        print(memory['observations'].shape)
        memory['sample_time'] = t_end - t_start

        return memory