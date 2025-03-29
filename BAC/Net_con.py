import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class A_net(nn.Module):
    def __init__(self, state_n, action_n, net_width):
        super(A_net, self).__init__()
        self.A = nn.Sequential(
            layer_init(nn.Linear(state_n, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, net_width)),
            nn.ReLU(),
        )
        self.mean = layer_init(nn.Linear(net_width, action_n))
        self.std = layer_init(nn.Linear(net_width, action_n))

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, s):
        p = self.A(s)
        mean = self.mean(p)
        std_log = torch.clip(self.std(p), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, std_log.exp()

    def get_distri(self, s):
        mean, std = self.forward(s)
        distri = Normal(mean, std)
        u = distri.rsample()
        a = torch.tanh(u)
        prob_log = distri.log_prob(u).sum(dim=-1, keepdim=True) - (torch.log(1 - a ** 2 + 1e-8)).sum(dim=-1, keepdim=True)

        return a, prob_log

    def get_action(self, s):
        u, _ = self.forward(s)
        return torch.tanh(u)


class Q_C_net(nn.Module):
    def __init__(self, state_n, action_n, net_width):
        super(Q_C_net, self).__init__()
        self.Q1_net = nn.Sequential(
            layer_init(nn.Linear(state_n + action_n, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, 1)),
            nn.Identity()
        )

        self.Q2_net = nn.Sequential(
            layer_init(nn.Linear(state_n + action_n, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, 1)),
            nn.Identity()
        )

    def forward(self, state, action):
        Input = torch.cat([state, action], -1)
        Q1 = self.Q1_net(Input)
        Q2 = self.Q2_net(Input)
        return Q1, Q2


class V_C_net(nn.Module):
    def __init__(self, state_n, net_width):
        super(V_C_net, self).__init__()
        self.V_net = nn.Sequential(
            layer_init(nn.Linear(state_n, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, 1)),
            nn.Identity()
        )

    def forward(self, state):
        V = self.V_net(state)
        return V
