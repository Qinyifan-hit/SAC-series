import torch
import torch.nn.functional as F
import numpy as np
import copy
from Net_con import A_net, Q_C_net, V_C_net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BAC_Agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.Actor = A_net(self.state_dim, self.action_dim, self.net_width).to(device)
        self.A_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.a_lr)

        self.Q_mix = Q_C_net(self.state_dim, self.action_dim, self.net_width).to(device)
        self.Q_optimizer = torch.optim.Adam(self.Q_mix.parameters(), lr=self.c_lr)
        self.Q_mix_target = copy.deepcopy(self.Q_mix)

        self.Value = V_C_net(self.state_dim, self.net_width).to(device)
        self.V_optimizer = torch.optim.Adam(self.Value.parameters(), lr=self.c_lr)

        self.expect_entropy = 0.6 * (-np.log(1 / self.action_dim))
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

    def action_select(self, s, iseval):
        with torch.no_grad():
            state = torch.FloatTensor(s).view(1, -1).to(device)
            if iseval:
                a = self.Actor.get_action(state)
            else:
                a, _ = self.Actor.get_distri(state)
            return a.cpu().numpy().squeeze(0)

    def train(self, Replay):
        s, a, r, s_, dw = Replay.sample(self.batch_size)
        self.alpha = self.log_alpha.exp().item()
        # Expectile regression
        _, log_prob_c = self.Actor.get_distri(s)
        V = self.Value(s) - self.alpha * log_prob_c
        with torch.no_grad():
            Q1, Q2 = self.Q_mix_target(s, a)
            Q = torch.min(Q1, Q2)
        QV_error = Q - V
        weight = torch.abs(self.quantile - (QV_error < 0).float())
        V_loss = (weight * QV_error.pow(2)).mean()
        self.V_optimizer.zero_grad()
        V_loss.backward()
        self.V_optimizer.step()

        # BEE
        with torch.no_grad():
            a_, log_prob_ = self.Actor.get_distri(s_)
            Q1_, Q2_ = self.Q_mix_target(s_, a_)
            Q_ = torch.min(Q1_, Q2_) - self.alpha * log_prob_
            Target_explore = r + self.gamma * Q_ * (~dw)
        Target_exploit = r + self.gamma * self.Value(s_) * (~dw)
        Target = self.lamda * Target_exploit + (1 - self.lamda) * Target_explore

        Q1, Q2 = self.Q_mix(s, a)
        Q_loss = F.mse_loss(Q1, Target) + F.mse_loss(Q2, Target)
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Actor update
        a, log_prob = self.Actor.get_distri(s)
        Q1, Q2 = self.Q_mix(s, a)
        Q = torch.min(Q1, Q2)
        A_loss = (self.alpha * log_prob - Q).mean()
        self.A_optimizer.zero_grad()
        A_loss.backward()
        self.A_optimizer.step()

        # alpha update
        with torch.no_grad():
            _, log_prob = self.Actor.get_distri(s)
            E_error = (-log_prob - self.expect_entropy).mean()
        alpha_loss = self.log_alpha.exp() * E_error
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p, p_target in zip(self.Q_mix.parameters(), self.Q_mix_target.parameters()):
            p_target.data.copy_(self.tua * p.data + (1 - self.tua) * p_target.data)

    def load(self, BName, ModelIdex):
        self.Actor.load_state_dict(torch.load("./model/{}_Actor{}.pth".format(BName, ModelIdex)))
        self.Q_mix.load_state_dict(torch.load("./model/{}_Q_mix{}.pth".format(BName, ModelIdex)))
        self.Value.load_state_dict(torch.load("./model/{}_Value{}.pth".format(BName, ModelIdex)))

        self.Q_mix_target = copy.deepcopy(self.Q_mix)

    def save(self, BName, ModelIdex):
        torch.save(self.Actor.state_dict(), "./model/{}_Actor{}.pth".format(BName, ModelIdex))
        torch.save(self.Q_mix.state_dict(), "./model/{}_Q_mix{}.pth".format(BName, ModelIdex))
        torch.save(self.Value.state_dict(), "./model/{}_Value{}.pth".format(BName, ModelIdex))


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
