import time
import os

import gym
import cv2
import numpy as np
import torch
import gym
import wandb

import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from tqdm.auto import trange
from collections import deque
from rlkit.buffer import ReplayBuffer, OnlineSampler
from rlkit.utils.wandb_logger import WandbLogger
from rlkit.policy import BasePolicy
from rlkit.nets import BaseEncoder

# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        eval_env_idx: int,
        logger: WandbLogger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        num_traj: int = 0,
        eval_episodes: int = 10,
        rendering: bool = False,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        buffer: ReplayBuffer = None,
        sampler: OnlineSampler = None,
        obs_dim: int = None,
        pomdp: list = None,
        import_policy: bool = False,
        device=None,
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.eval_env_idx = eval_env_idx
        self.buffer = buffer
        self.sampler = sampler
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._num_traj = num_traj
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.obs_dim = obs_dim[0]        
        self.pomdp = pomdp

        self._device = device

        self.current_epoch = 0
        self.log_interval = 10
        self.rendering = rendering
        self.recorded_frames = []

        if import_policy:
            print('...loading previous model')
            self.policy.load_state_dict(torch.load('model/model.pth'))
            self.policy.eval()

    def train(self, seed) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in trange(self._epoch, desc=f"Epoch"):
            self.current_epoch = e
            self.policy.train()

            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                batch = self.buffer.sample(self._batch_size, self._num_traj)
                loss = self.policy.learn(batch)
                self.logger.store(**loss)
                self.logger.write_without_reset(int(e*self._step_per_epoch + it))

                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate(seed)
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_success_mean, ep_success_std = np.mean(eval_info["eval/episode_success"]), np.std(eval_info["eval/episode_success"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            
            eval_data = {"eval/episode_reward": ep_reward_mean,
                         "eval/episode_reward_std": ep_reward_std,
                         "eval/ep_success_mean": ep_success_mean,
                         "eval/ep_success_std": ep_success_std,
                         "eval/episode_length": ep_length_mean,
                         "eval/episode_length_std": ep_length_std
                         }
            try:
                norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                last_10_performance.append(norm_ep_rew_mean)
                norm_data = {
                    "eval/normalized_episode_reward_std": norm_ep_rew_std,
                    "eval/normalized_episode_reward": norm_ep_rew_mean,
                }
                eval_data.update(norm_data)
            except:
                last_10_performance.append(ep_reward_mean)
            self.logger.store(**eval_data)        
            self.logger.write(int(e*self._step_per_epoch + it), display=False)
            if self.current_epoch % self.log_interval == 0:
                # save checkpoint
                torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy_" +str(e)+ ".pth"))

        self.logger.print("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.log_dir, "policy.pth"))
        return {"last_10_performance": np.mean(last_10_performance)}
    
    def online_train(self, seed) -> Dict[str, float]:
        start_time = time.time()

        last_10_performance = deque(maxlen=10)
        # train loop
        for e in trange(self._epoch, desc=f"Epoch"):
            self.current_epoch = e
            self.policy.train()
            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                batch = self.sampler.collect_samples(self.train_env, self.policy, seed)
                loss = self.policy.learn(batch); loss['sample_time'] = batch['sample_time']
                self.logger.store(**loss)
                self.logger.write_without_reset(int(e*self._step_per_epoch + it))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate(seed)
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_success_rate_mean, ep_success_rate_std = np.mean(eval_info["eval/episode_success_rate"]), np.std(eval_info["eval/episode_success_rate"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])

            eval_data = {"eval/episode_reward": ep_reward_mean,
                         "eval/episode_reward_std": ep_reward_std,
                         "eval/ep_success_mean": ep_success_rate_mean,
                         "eval/ep_success_std": ep_success_rate_std,
                         "eval/episode_length": ep_length_mean,
                         "eval/episode_length_std": ep_length_std
                         }
            try:
                norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                last_10_performance.append(norm_ep_rew_mean)
                norm_data = {
                    "eval/normalized_episode_reward_std": norm_ep_rew_std,
                    "eval/normalized_episode_reward": norm_ep_rew_mean,
                }
                eval_data.update(norm_data)
            except:
                last_10_performance.append(ep_reward_mean)
            self.logger.store(**eval_data)        
            self.logger.write(int(e*self._step_per_epoch + it), display=False)
            if self.current_epoch % self.log_interval == 0:
                # save checkpoint
                torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy_" +str(e)+ ".pth"))
        
        self.logger.print("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.log_dir, "policy.pth"))
        return {"last_10_performance": np.mean(last_10_performance)}
    
    def normalize_obs(self, obs):
        if self.sampler is not None: # check if it is online training
            if self.sampler.running_state is not None: # check if there is running state enabled
                self.sampler.running_state.fix = True
                obs = self.sampler.running_state(obs)
                self.sampler.running_state.fix = False
        elif self.buffer is not None: # check if it is offline training
            if self.buffer._obs_normalized: # check if obs is normalized
                obs = (obs - self.buffer.obs_mean) / (self.buffer.obs_std + 1e-10)
        return obs
    
    def _evaluate(self, seed) -> Dict[str, List[float]]:
        self.policy.eval()
        try:
            obs, _ = self.eval_env.reset(seed=seed)
        except:
            obs = self.eval_env.reset(seed=seed)
        obs = self.normalize_obs(obs)
        action = np.zeros((np.prod(self.eval_env.action_space.shape), ))
        next_obs = obs # initialization
        reward = 0.0
        mask = 1.0
        
        with torch.no_grad():
            obs, _, e_obs, _ = self.policy.encode_obs((obs, action, next_obs, [reward], mask), env_idx=self.eval_env_idx)

        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length, episode_success = 0, 0, 0

        while num_episodes < self._eval_episodes:
            with torch.no_grad():
                action = self.policy.select_action(e_obs, deterministic=True) #(obs).reshape(1,-1)
            try:
                next_obs, reward, trunc, terminal, infos = self.eval_env.step(action.flatten())
                done = terminal or trunc
            except:
                next_obs, reward, terminal, infos = self.eval_env.step(action.flatten())
                done = terminal
            
            if self.rendering:
                if num_episodes == 0:
                    self.recorded_frames.append(self.eval_env.render())
            
            episode_reward += reward
            try:
                episode_success += infos['success']
            except:
                episode_success += 0.0
            episode_length += 1
            
            next_obs = self.normalize_obs(next_obs)
            with torch.no_grad():
                _, next_obs, _, e_next_obs = self.policy.encode_obs((obs, action, next_obs, [reward], mask), env_idx=self.eval_env_idx, reset=False)
            
            obs = next_obs
            e_obs = e_next_obs

            if done:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length, "episode_success_rate":episode_success/episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                try:
                    obs, _ = self.eval_env.reset(seed=seed)
                except:
                    obs = self.eval_env.reset(seed=seed)

        if self.current_epoch % self.log_interval == 0:
            if self.rendering:
                self.save_rendering(self.logger.checkpoint_dir)
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval/episode_success_rate": [ep_info["episode_success_rate"] for ep_info in eval_ep_info_buffer],
        }
    
    def save_rendering(self, path):
        directory = os.path.join(path, 'video')
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = 'rendering' + str(self.current_epoch) +'.avi'
        output_file = os.path.join(directory, file_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI file
        fps = 120
        width = 480
        height = 480
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for frame in self.recorded_frames:
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()
        self.recorded_frames = []


