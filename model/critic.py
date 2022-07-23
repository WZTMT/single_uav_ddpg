import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, init_w=3e-3):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(n_states + n_actions, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 512)
        self.linear5 = nn.Linear(512, 1)
        # 随机初始化为较小的值
        nn.init.uniform_(self.linear5.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.linear5.bias.detach(), a=-init_w, b=init_w)

        # 另一种写法
        # self.linear5.weight.data.uniform_(-init_w, init_w)
        # self.linear5.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # 按维数1拼接(按维数1拼接为横着拼，按维数0拼接为竖着拼)
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        q = self.linear3(x)
        return q
