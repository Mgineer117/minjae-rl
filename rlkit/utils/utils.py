import random
import os
import torch
import numpy as np

from typing import Any, DefaultDict, Dict, List, Optional, Tuple

def select_device(gpu_idx=0):
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda:'+str(gpu_idx)) 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")
    return device

def seed_all(seed=0, others: Optional[list] = None):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if others is not None:
        if hasattr(others, "seed"):
            others.seed(seed)
            return True
        try:
            for item in others:
                if hasattr(item, "seed"):
                    item.seed(seed)
        except:
            pass

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
        
def estimate_advantages(rewards, masks, values, gamma, tau, device):
    rewards, masks, values = rewards.to(torch.device('cpu')), masks.to(torch.device('cpu')), values.to(torch.device('cpu'))
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages, returns = advantages.to(device), returns.to(device)
    return advantages, returns

def estimate_constraint_value(costs, masks, gamma, device):
    costs, masks = costs.to(torch.device('cpu')), masks.to(torch.device('cpu'))
    tensor_type = type(costs)
    constraint_value = torch.tensor(0)

    j = 1
    traj_num = 1
    for i in range(costs.size(0)):
        constraint_value = constraint_value + costs[i] * gamma**(j-1)

        if masks[i] == 0:
            j = 1 #reset
            traj_num = traj_num + 1
        else: 
            j = j+1
            
    constraint_value = constraint_value/traj_num
    constraint_value = constraint_value.to(device)
    return constraint_value[0]