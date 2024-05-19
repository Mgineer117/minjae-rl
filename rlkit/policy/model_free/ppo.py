import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import math
import time

from typing import Dict, Union, Tuple
from rlkit.policy import BasePolicy
from rlkit.nets import BaseEncoder
from rlkit.utils.utils  import estimate_advantages, get_flat_params_from, set_flat_params_to

class PPOPolicy(BasePolicy):
    def __init__(
            self, 
            actor: nn.Module, 
            actor_optim: torch.optim.Optimizer,
            critic: nn.Module,  
            critic_optim: torch.optim.Optimizer,
            encoder: BaseEncoder = None,
            encoder_optim: torch.optim.Optimizer = None,
            tau: float = 0.95,
            gamma: float = 0.99,
            K_epochs: int = 3,
            eps_clip: float = 0.2,
            l2_reg: float = 1e-4,
            device = None
            ):
        super().__init__()

        self.actor = actor
        self.actor_optim = actor_optim
        self.critic = critic
        self.critic_optim = critic_optim
        self.encoder = encoder
        self.encoder_optim = encoder_optim

        self.loss_fn = torch.nn.MSELoss()

        self._gamma = gamma
        self._tau = tau
        self._K_epochs = K_epochs
        self._eps_clip = eps_clip
        self._l2_reg = l2_reg

        self.param_size = sum(p.numel() for p in self.actor.parameters())
        self.device = device
    
    def train(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        reset: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        logprob = dist.log_prob(action)
        return action, logprob

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actforward(obs, deterministic)
        return action.cpu().numpy()

    def encode_obs(self, mdp_tuple, running_state=None, env_idx = None, reset=True):
        '''
        Given mdp = (s, a, s', r, mask)
        return embedding, embedded_next_obs = embedding is attached to the next_ob
          since it should include the information of reward and transition dynamics
        '''
        obs, actions, next_obs, rewards, masks = mdp_tuple
        if running_state is not None:
            obs = running_state(obs)
            next_obs = running_state(next_obs)
        # check dimension
        is_batch = True if len(obs.shape) > 1 else False
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        masks = torch.as_tensor(masks, device=self.device, dtype=torch.float32)

        if self.encoder.encoder_type == 'none':
            embedding = None
            embedded_obs = obs
            embedded_next_obs = next_obs
        elif self.encoder.encoder_type == 'onehot':
            obs_embedding = self.encoder(obs, env_idx)
            next_obs_embedding = self.encoder(next_obs, env_idx)
            embedded_obs = torch.concatenate((obs_embedding, obs), axis=-1) 
            embedded_next_obs = torch.concatenate((next_obs_embedding, next_obs), axis=-1)
        elif self.encoder.encoder_type == 'recurrent':
            if is_batch:
                obs = torch.concatenate((obs[0][None, :], obs), axis=0)
                actions = torch.concatenate((actions[0][None, :], actions), axis=0)
                next_obs = torch.concatenate((obs[0][None, :], next_obs), axis=0)
                rewards = torch.concatenate((torch.tensor([0.0]).to(self.device), rewards), axis=0)
                masks = torch.concatenate((torch.tensor([1.0]).to(self.device), masks), axis=0)

                padded_mdp = self.pack4rnn(obs, actions, next_obs, rewards, masks)
                embedding = self.encoder(padded_mdp, padded=True)
                embedded_obs = torch.concatenate((embedding[:-1], obs), axis=-1)
                embedded_next_obs = torch.concatenate((embedding[1:], next_obs), axis=-1)
            else:
                mdp = torch.concatenate((obs, actions, next_obs, rewards), axis=-1)
                mdp = mdp[None, None, :]
                embedding = self.encoder(mdp, do_reset=reset)
                embedded_next_obs = torch.concatenate((embedding, next_obs), axis=-1)
                embedded_obs = embedded_next_obs

        return obs, next_obs, embedded_obs, embedded_next_obs

    def pack4rnn(self, obss, actions, next_obss, rewards, masks):
        trajs = []
        prev_i = 0
        rewards = rewards[:, None]
        for i, mask in enumerate(masks):
            if mask == 0:
                trajs.append(torch.concatenate((obss[prev_i:i+1, :], actions[prev_i:i+1, :], next_obss[prev_i:i+1, :], rewards[prev_i:i+1, :]), axis=-1))
                prev_i = i + 1    
        # Step 1: Pad the sequences
        padded_data = pad_sequence(trajs, batch_first=True)  # (batch_size, max_seq_len, 24)
        sequence_lengths = torch.tensor([x.size(0) for x in trajs]) 

        # Step 2: Pack the padded sequences
        packed_data = pack_padded_sequence(padded_data, sequence_lengths, batch_first=True, enforce_sorted=False)
        print(packed_data)
        print(packed_data.shape)

        return packed_data
    
    def learn(self, batch):
        obss, actions, next_obss, rewards, masks, logprobs, env_idxs, successes = \
            batch["observations"], batch["actions"], batch["next_observations"], \
                batch["rewards"], batch["masks"], batch["logprobs"], batch["env_idxs"], batch["successes"]
        
        mdp_tuple = (obss, actions, next_obss, rewards, masks)

        '''Update the parameters'''
        for _ in range(self._K_epochs):
            _, _, embedded_obss, _ = self.encode_obs(mdp_tuple, env_idx=env_idxs)

            r_pred = self.critic(embedded_obss)

            """get advantage estimation from the trajectories"""
            advantages, returns = estimate_advantages(rewards, masks, r_pred.detach(), self._gamma, self._tau, self.device)

            '''get policy outpu'''
            dist = self.actor(embedded_obss.detach())
            new_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            
            '''
            if self.encoder_optim is not None:
                encoder_loss = -torch.mean(r_pred)

                self.encoder_optim.zero_grad()
                encoder_loss.backward(retain_graph=True)            
                self.encoder_optim.step()

            v_loss = self.loss_fn(r_pred, returns)
            
            #self.critic_optim.zero_grad()
            v_loss.backward()
            self.critic_optim.step()
            '''
            v_loss = self.loss_fn(r_pred, returns)

            if self.encoder_optim is not None:
                self.critic_optim.zero_grad(); self.encoder_optim.zero_grad()
                v_loss.backward()
                self.critic_optim.step(); self.encoder_optim.step()
            else:
                self.critic_optim.zero_grad()
                v_loss.backward()
                self.critic_optim.step()

            ratios = torch.exp(new_logprobs - logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self._eps_clip, 1+self._eps_clip) * advantages

            loss = torch.mean(-torch.min(surr1, surr2) + 0.5 * v_loss.detach() - 0.01 * dist_entropy)

            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()


        result = {
            'loss/critic_loss': v_loss.item(),
            'loss/actor_loss': loss.item(),
            'train/stochastic_reward': rewards.mean().item(),
            'train/success': successes.mean().item()
        }
        
        return result 