import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer


class DDPG:
    def __init__(self, n_states, n_actions, cfg):
        super().__init__()
        self.device = cfg.device
        self.critic = Critic(n_states, n_actions).to(self.device)
        self.actor = Actor(n_states, n_actions).to(self.device)
        self.target_critic = Critic(n_states, n_actions).to(self.device)
        self.target_actor = Actor(n_states, n_actions).to(self.device)

        # 复制参数到目标网络
        # param是online网络的参数，tensor1.copy_(tensor2)，将2的元素复制给1
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau  # 软更新率
        self.gamma = cfg.gamma  # 折扣率

    '''
    FloatTensor()将np.ndarray转为tensor
    unsqueeze()在指定的位置上增加1维的维度，如(2,3).unsqueeze(0)后变成(1,2,3)
    '''
    def choose_action(self, state):
        # 变成二维tensor，[1,3]，因为一维的标量不能做tensor的乘法，actor中第一层的weight形状为[3,512]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        # tensor.detach()与tensor.data的功能相同，但是若因外部修改导致梯度反向传播出错，.detach()会报错，.data不行
        action = action.detach().cpu().numpy()
        ax = action[0, 0]  # 前面是一个数组，[0,0]取出第一个元素
        ay = action[0, 1]
        az = action[0, 2]
        return ax, ay, az

    def update(self):
        if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放缓冲池中随机采样一个批量的transition
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))  # 疑问：一维二维都能作为网络输入？
        policy_loss = -policy_loss.mean()

        # 用target网络计算y值(expected_value)
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        mse_loss = nn.MSELoss()
        value_loss = mse_loss(value, expected_value.detach())

        # online网络更新
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # target网络更新，软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')  # 后缀.pt和.pth没什么区别

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))


if __name__ == '__main__':
    state = torch.FloatTensor([.1]).mean()
    print(state)
