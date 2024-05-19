import numpy as np

import torch
import torch.nn as nn
import math
import time

from typing import Dict, Union, Tuple
from rlkit.policy import BasePolicy
from rlkit.utils.utils  import estimate_advantages, get_flat_params_from, set_flat_params_to

def conjugate_gradients(Avp, b, nsteps, device, residual_tol=1e-10):
    x = torch.zeros(b.size()).to(device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=15, accept_ratio=0.1):
        fval = f(True).item()
        
        for stepfrac in [.5**x for x in range(max_backtracks)]:
            x_new = x + stepfrac * fullstep
            set_flat_params_to(model, x_new)
            fval_new = f(True).item()

            actual_improve = fval - fval_new

            expected_improve = expected_improve_full * stepfrac

            ratio = actual_improve / expected_improve

            if actual_improve > 0 and ratio > accept_ratio:
                return True, x_new
        return False, x

class TRPOPolicy(BasePolicy):
    def __init__(
            self, 
            actor: nn.Module, 
            critic: nn.Module,  
            critic_optim: torch.optim.Optimizer,
            tau: float = 0.95,
            gamma: float  = 0.99,
            max_kl: float = 1e-3,
            damping: float = 1e-2,
            l2_reg: float = 1e-4,
            device = None
            ):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.critic_optim = critic_optim

        self.loss_fn = torch.nn.MSELoss()

        self._gamma = gamma
        self._tau = tau
        self._max_kl = max_kl
        self._damping = damping
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
        deterministic: bool = False
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

    def learn(self, batch):
        obss, actions, rewards, masks, successes = \
            batch["observations"], batch["actions"], batch["rewards"], batch["masks"], batch["successes"]
        #successes = torch.sum(successes) / len(torch.where(masks == 0.0)[0])

        with torch.no_grad():
            values = self.critic(obss)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self._gamma, self._tau, self.device)

        """update critic"""
        def closure():
            self.critic_optim.zero_grad()
            r_pred = self.critic(obss)
            v_loss = self.loss_fn(r_pred, returns)
            for param in self.critic.parameters():
                v_loss += param.pow(2).sum() * self._l2_reg
            v_loss.backward()
            return v_loss
        
        self.critic_optim.step(closure)

        with torch.no_grad():
            r_pred = self.critic(obss)

        v_loss = self.loss_fn(r_pred, returns)

        """update policy"""
        with torch.no_grad():
            dist = self.actor(obss)
            fixed_log_probs = dist.log_prob(actions)

        def get_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                dist = self.actor(obss)
                log_probs = dist.log_prob(actions)
                action_loss = -advantages * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()
            
        """directly compute Hessian*vector from KL"""
        def Fvp_direct(v):
            kl = self.actor.get_kl(obss)
            kl = kl.mean()

            grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * v).sum()
            grads = torch.autograd.grad(kl_v, self.actor.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

            return flat_grad_grad_kl + v * self._damping

        Fvp = Fvp_direct
        
        loss = get_loss()
        grads = torch.autograd.grad(loss, self.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        stepdir = conjugate_gradients(Fvp, -loss_grad, 10, device=self.device)

        shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
        lm = math.sqrt(self._max_kl / shs)
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)

        prev_params = get_flat_params_from(self.actor)
        ln_sch_success, new_params = line_search(self.actor, get_loss, prev_params, fullstep, expected_improve)
        set_flat_params_to(self.actor, new_params)

        result = {
            'loss/critic_loss': v_loss.item(),
            'loss/actor_loss': loss.item(),
            'train/stochastic_reward': rewards.mean().item(),
            'train/success': successes.item()
        }
        
        return result 