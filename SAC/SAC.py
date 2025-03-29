import torch
import torch.nn.functional as F
import numpy as np
import copy
from Net_con import A_net, C_net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SAC_Agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.Actor = A_net(self.action_dim, self.state_dim, self.net_width).to(device)
        self.A_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.a_lr)

        self.Critic1 = C_net(self.action_dim, self.state_dim, self.net_width).to(device)
        self.C1_optimizer = torch.optim.Adam(self.Critic1.parameters(), lr=self.c_lr)
        self.Critic2 = C_net(self.action_dim, self.state_dim, self.net_width).to(device)
        self.C2_optimizer = torch.optim.Adam(self.Critic2.parameters(), lr=self.c_lr)

        self.C1_target = copy.deepcopy(self.Critic1)
        self.C2_target = copy.deepcopy(self.Critic2)

        self.expect_entropy = 0.6 * (-np.log(1 / self.action_dim))
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.a_lr)

    def action_select(self, s, iseval):
        s = torch.FloatTensor(s).view(1, -1).to(device)
        with torch.no_grad():
            if iseval:
                a = self.Actor.get_action(s)
            else:
                a, _ = self.Actor.get_distri(s)
            return a.cpu().squeeze(0).numpy()

    def train(self, Replay):
        s, a, r, s_, dw = Replay.sample(self.batch_size)
        self.alpha = self.log_alpha.exp().item()
        with torch.no_grad():
            a_, prob_log_ = self.Actor.get_distri(s_)
            Q1_ = self.C1_target(a_, s_)
            Q2_ = self.C2_target(a_, s_)
            Q_ = torch.min(Q1_, Q2_)
            V_ = Q_ - self.alpha * prob_log_
            Q_target = r + (~dw) * self.gamma * V_

        Q1 = self.Critic1(a, s)
        C1_loss = F.mse_loss(Q1, Q_target)
        self.C1_optimizer.zero_grad()
        C1_loss.backward()
        self.C1_optimizer.step()

        Q2 = self.Critic2(a, s)
        C2_loss = F.mse_loss(Q2, Q_target)
        self.C2_optimizer.zero_grad()
        C2_loss.backward()
        self.C2_optimizer.step()

        a, prob_log = self.Actor.get_distri(s)
        Q1 = self.Critic1(a, s)
        Q2 = self.Critic2(a, s)
        Q = torch.min(Q1, Q2)
        A_loss = (self.alpha * prob_log - Q).mean()
        self.A_optimizer.zero_grad()
        A_loss.backward()
        self.A_optimizer.step()

        with torch.no_grad():
            _, prob_log = self.Actor.get_distri(s)
            E_error = (-prob_log - self.expect_entropy).mean()
        alpha_loss = self.log_alpha.exp() * E_error
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p, p_target in zip(self.Critic1.parameters(), self.C1_target.parameters()):
            p_target.data.copy_(self.tua * p.data + (1 - self.tua) * p_target.data)

        for p, p_target in zip(self.Critic2.parameters(), self.C2_target.parameters()):
            p_target.data.copy_(self.tua * p.data + (1 - self.tua) * p_target.data)

    def load(self, Env_name, Index):
        self.Actor.load_state_dict(torch.load("./model/{}_Actor{}.pth".format(Env_name, Index), map_location=device))
        self.Critic1.load_state_dict(
            torch.load("./model/{}_Critic1{}.pth".format(Env_name, Index), map_location=device))
        self.Critic2.load_state_dict(
            torch.load("./model/{}_Critic2{}.pth".format(Env_name, Index), map_location=device))
        self.C1_target = copy.deepcopy(self.Critic1)
        self.C2_target = copy.deepcopy(self.Critic2)

    def save(self, Env_name, Index):
        torch.save(self.Actor.state_dict(), "./model/{}_Actor{}.pth".format(Env_name, Index))
        torch.save(self.Critic1.state_dict(), "./model/{}_Critic1{}.pth".format(Env_name, Index))
        torch.save(self.Critic2.state_dict(), "./model/{}_Critic2{}.pth".format(Env_name, Index))


class Buffer_Replay(object):
    def __init__(self, state_n, action_n, max_size):
        self.max_size = int(max_size)
        self.Ind = int(0)
        self.s = np.zeros((self.max_size, state_n), dtype=np.float32)
        self.s_ = copy.deepcopy(self.s)
        self.r = np.zeros((self.max_size, 1), dtype=np.float32)
        self.a = np.zeros((self.max_size, action_n), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.bool_)
        self.dw = copy.deepcopy(self.done)
        self.size = int(0)

    def add(self, s, a, r, s_, done, dw):
        Ind = self.Ind
        self.a[Ind] = a
        self.s[Ind] = s
        self.s_[Ind] = s_
        self.r[Ind] = r
        self.done[Ind] = done
        self.dw[Ind] = dw
        self.Ind = (self.Ind + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, b_size):
        Ind = np.random.choice(self.size, b_size, replace=False)
        return (
            torch.FloatTensor(self.s[Ind]).to(device),
            torch.FloatTensor(self.a[Ind]).to(device),
            torch.FloatTensor(self.r[Ind]).to(device),
            torch.FloatTensor(self.s_[Ind]).to(device),
            torch.BoolTensor(self.dw[Ind]).to(device),
        )
