import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from typing import Optional, Union, Tuple, Dict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        obs_norm: bool = False,
        rew_norm: bool = False,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype
        self.obs_norm = obs_norm
        self.rew_norm = rew_norm

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)
        initial_observations = np.array(next_observations[(terminals == 1).squeeze()], dtype=self.obs_dtype)
        try:
            masks = np.array(dataset["masks"], dtype=np.int32).reshape(-1, 1)
            env_idxs = np.array(dataset["env_idxs"], dtype=np.int32).reshape(-1, 1)    
        except:
            masks = np.int32(~terminals.astype(bool)).reshape(-1, 1)    
            env_idxs = np.zeros(masks.shape, dtype=np.int32).reshape(-1, 1)    

        self.observations = observations
        self.next_observations = next_observations
        self.initial_observations = initial_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.masks = masks
        self.env_idxs = env_idxs

        if self.obs_norm:
            self.normalize_obs()
        if self.rew_norm:
            self.normalize_rew()

        self._ptr = len(observations)
        self._size = len(observations)
        self._init_size = len(initial_observations)
        self._tot_traj = terminals.sum()
        self._traj_indexes = np.array([0] + list(np.where(self.terminals == 1)[0] + 1))
    
    def normalize_obs(self, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(self.observations, axis=0)
        std = np.std(self.observations, axis=0) 
        self.observations = (self.observations - mean) / (std + eps)
        self.next_observations = (self.next_observations - mean) / (std + eps)
        self.initial_observations = (self.initial_observations - mean) / (std + eps)

        self.obs_mean, self.obs_std = mean, std
    
    def normalize_rew(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.rewards.mean()
        std = self.rewards.std() + eps
        self.rewards = (self.rewards - mean) / std
        self.rew_mean, self.rew_std = mean, std

    def sample(self, batch_size: int = 512, num_traj: int = 0) -> Dict[str, torch.Tensor]:
        if num_traj == 0:
            batch_indexes = np.random.randint(0, self._size, size=batch_size)
            init_batch_indexes = np.random.randint(0, self._init_size, size=batch_size)
            batch = {
                "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
                "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
                "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
                "initial_observations": torch.tensor(self.initial_observations[init_batch_indexes]).to(self.device),
                "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
                "masks": torch.tensor(self.masks[batch_indexes]).to(self.device),
                "env_idxs": torch.tensor(self.masks[batch_indexes]).to(self.device),
                "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
            }
        else:
            traj_indexes = np.random.randint(0, self._tot_traj, size=num_traj)
            batch_indexes = np.concatenate([np.arange(self._traj_indexes[traj_idx], self._traj_indexes[traj_idx + 1]) 
                                            for traj_idx in traj_indexes])
            batch = {
                "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
                "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
                "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
                "initial_observations": torch.tensor(self.initial_observations[traj_indexes]).to(self.device),
                "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
                "masks": torch.tensor(self.masks[batch_indexes]).to(self.device),
                "env_idxs": torch.tensor(self.env_idxs[batch_indexes]).to(self.device),
                "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
            }

        return batch
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "initial_observations": self.initial_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }
