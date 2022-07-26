import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from model.actor import Actor
from model.critic import Critic
from model.replay_buffer import ReplayBuffer


class DDPG:
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.critic = Critic(cfg.n_state, cfg.n_action).to(self.device)
        self.actor = Actor(cfg.n_state, cfg.n_action).to(self.device)
        self.target_critic = Critic(cfg.n_state, cfg.n_action).to(self.device)
        self.target_actor = Actor(cfg.n_state, cfg.n_action).to(self.device)

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
        # 变成二维tensor，[1,3]，因为一维的标量不能做tensor的乘法，actor中第一层的weight形状为[3,512](标量也可以做乘法)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        # tensor.detach()与tensor.data的功能相同，但是若因外部修改导致梯度反向传播出错，.detach()会报错，.data不行，且.detach()得到的数据不带梯度
        action = action.detach().cpu().squeeze(0).numpy()
        return action

    def update(self):
        if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放缓冲池中随机采样一个批量的transition，每次用一个batch_size的数据进行训练
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量，得到的都是二维的tensor，第一个数字是batch_size，state(256,47)，reward(256,1)
        state = torch.FloatTensor(np.array(state)).to(self.device)  # 计算出来直接是一个二维的tensor
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))  # 一维二维都能作为网络输入, 输出与输入的维度保持一致
        policy_loss = -policy_loss.mean()  # 计算一个bach_size的policy_loss的均值

        # 用target网络计算y值(expected_value)
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())  # 一个网络的输出作为另一个网络的输入，需要.detach()，取出不带梯度的数据
        # y值为当前的奖励加上未来可能的Q值，而对于episode结束的Transition，因为不存在未来的动作和状态，所以未来部分为0
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)  # 将数据裁剪到min和max之间

        value = self.critic(state, action)  # online网络的计算结果为原值
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
    reward = np.array([.5])
    state = torch.FloatTensor(reward).unsqueeze(0)
    print(state)

    input = torch.FloatTensor([[.2, .3, .4], [.4, .5, .6]])
    print(input.shape)
    test1 = nn.Linear(3, 1024)
    test2 = nn.Linear(1024, 1)
    x = test1(input)
    output = test2(x)
    print(output.shape)
    print(output)
    print(output.detach())
    print(output.mean())
