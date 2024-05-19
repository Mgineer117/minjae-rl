import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


def identity(x):
    return x

class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation=Swish(),
        layer_norm=True,
        with_residual=True,
        dropout=0.1
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.with_residual = with_residual
    
    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.dropout is not None:
            y = self.dropout(y)
        if self.with_residual:
            y = x + y
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        return y


class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        mlp_initialization = False,
        hidden_dims=[200, 200, 200, 200],
        rnn_num_layers=3,
        dropout_rate=0.1,
        device="cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.device = torch.device(device)

        self.activation = Swish()
        self.rnn_layer = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=rnn_num_layers,
            batch_first=True
        )
        module_list = []
        self.input_layer = ResBlock(input_dim, hidden_dims[0], dropout=dropout_rate, with_residual=False)
        dims = list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            module_list.append(ResBlock(in_dim, out_dim, dropout=dropout_rate))
        self.backbones = nn.ModuleList(module_list)
        self.merge_layer = nn.Linear(dims[0] + dims[-1], hidden_dims[0])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.to(self.device)

    def forward(self, input, h_state=None):
        batch_size, num_timesteps, _ = input.shape
        input = torch.as_tensor(input, dtype=torch.float32).to(self.device)
        rnn_output, h_state = self.rnn_layer(input, h_state)
        rnn_output = rnn_output.reshape(-1, self.hidden_dims[0])
        input = input.view(-1, self.input_dim)
        output = self.input_layer(input)
        output = torch.cat([output, rnn_output], dim=-1)
        output = self.activation(self.merge_layer(output))
        for layer in self.backbones:
            output = layer(output)
        output = self.output_layer(output)
        output = output.view(batch_size, num_timesteps, -1)
        return output, h_state


class RecurrentEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_sizes: Union[List[int], Tuple[int]],
            output_size,
            rnn_initialization: bool = True,
            activation: nn.Module = nn.ReLU,
            output_activation=identity,
            dropout_rate: Optional[float] = None,
            device="cpu"
    ):
        super().__init__()
        self.rnn_hidden_dim = hidden_sizes[-1]
        self.embed_dim = output_size

        hidden_sizes = [input_size] + list(hidden_sizes)
        model = []
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            linear_layer = nn.Linear(in_dim, out_dim)
            if rnn_initialization:
                nn.init.xavier_uniform_(linear_layer.weight)
                linear_layer.bias.data.fill_(0.01)
            model += [linear_layer, activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]
        self.fcs = nn.Sequential(*model).to(device)

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.rnn_hidden_dim, self.rnn_hidden_dim, num_layers=1, batch_first=True).to(device)

        self.last_layer = nn.Linear(self.rnn_hidden_dim, output_size).to(device)
        if rnn_initialization:
            nn.init.uniform_(self.last_layer.weight, a=-3e-3, b=3e-3)  # Set weights using uniform initialization
            nn.init.uniform_(self.last_layer.bias, a=-3e-3, b=3e-3)  # Set weights using uniform initialization

        self.output_activation = output_activation
        self.encoder_type = 'recurrent'
        self.device = torch.device(device)

        self.reset()

    def forward(self, input, do_reset=True, padded=False):
        input = torch.as_tensor(input, device=self.device, dtype=torch.float32)
        
        # expects inputs of dimension (task, seq, feat)
        trj, seq, feat = input.shape
        out = input.reshape(trj * seq, feat)

        # embed with MLP
        out = self.fcs(out)

        out = out.view(trj, seq, -1)
        
        if do_reset:
            self.reset()
        out, (hn, cn) = self.lstm(out, (self.hn, self.cn))
        self.hn = hn
        self.cn = cn
        # take the last hidden state to predict z
        #out = out[:, -1, :]

        # output layer
        preactivation = self.last_layer(out)
        embedding = self.output_activation(preactivation)

        if padded:
            embedding, _ = pad_packed_sequence(embedding, batch_first=True)
        
        return embedding.squeeze()

    def reset(self, num_tasks=1):
        self.hn = torch.zeros(num_tasks, 1, self.rnn_hidden_dim).to(self.device)
        self.cn = torch.zeros(num_tasks, 1, self.rnn_hidden_dim).to(self.device)
    
if __name__ == "__main__":
    model = RNNModel(14, 12)
    x = torch.randn(64, 20, 14)
    y, _ = model(x)
    print(y.shape)