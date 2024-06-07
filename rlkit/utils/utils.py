import random
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

from rlkit.nets import MLP, RNNModel, RecurrentEncoder, BaseEncoder, OneHotEncoder

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
    #torch.use_deterministic_algorithms(True)
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

def estimate_episodic_value(costs, masks, gamma, device):
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

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def visualize_latent_variable(tasks_name, latent_data, latent_path):
    # Define the task name list
    num_tasks = len(tasks_name)

    if num_tasks > 5:
        col = 2
        row = int(num_tasks/col)
        tasks = [tasks_name[i:i+row] for i in range(0, num_tasks, row)]

    else:
        col = 1
        row = num_tasks
        tasks = [tasks_name]

    data_per_task = 500
    data_dimensions = latent_data[0].shape[-1]
    highlight_interval = 25

    # Generate random data for each task (500, 5) for each task
    data = {task_name: latent_data[i][:data_per_task, :] for i, task_name in enumerate(tasks_name)}

    # Sample every 25th index
    sampled_indices = np.arange(0, data_per_task, highlight_interval)
    num_sampled_points = len(sampled_indices) 

    # Set up the plot with a calculated figure size to ensure square blocks
    fig_width = num_sampled_points * col  # width in "blocks"
    block_size = 1  # each block is a 1x1 square in figure units
    fig_height = row * data_dimensions * block_size

    fig, axs = plt.subplots(row, col, figsize=(fig_width * block_size, fig_height), squeeze=False, sharex=True)

    # Plot each task's data
    for col, task_list in enumerate(tasks):
        for idx, task in enumerate(task_list):
            task_data = data[task]

            # Sample every 25th index
            sampled_data = task_data[sampled_indices]

            # Transpose the data to match the expected heatmap format
            sampled_data_transposed = sampled_data.T

            # Create a heatmap-like plot
            cax = axs[idx, col].imshow(sampled_data_transposed, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
            axs[idx, col].set_title(task, fontsize=32, fontweight='bold')
            axs[idx, col].tick_params(axis='both', which='both', labelsize=18)
            axs[idx, col].set_yticks([])

            # Display numeric values on the heatmap
            for i in range(sampled_data_transposed.shape[1]):
                for j in range(sampled_data_transposed.shape[0]):
                    value = sampled_data[i, j]
                    text = axs[idx, col].text(i, j, f'{value:.2f}', ha='center', va='center', color='black', fontsize=20, fontweight='bold')
                    text.set_path_effects([withStroke(linewidth=2, foreground='white')])

            # Add a color bar to the side
            cbar = fig.colorbar(cax, ax=axs[idx, col])
            cbar.ax.tick_params(labelsize=26)  # Enlarge the ticks and tick labels of the color bar

    # Set the global x-axis labels
    time_intervals = sampled_indices
    for ax in axs[-1, :]:
        ax.set_xticks(np.arange(len(time_intervals)))
        ax.set_xticklabels(time_intervals, fontsize=24)
        ax.set_xlabel('Time', fontsize=28, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(latent_path)