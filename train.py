import os
import sys

import datetime

import numpy as np
import torch
import argparse
import airsim

from torch.utils.tensorboard import SummaryWriter
from model.ou_noise import OrnsteinUhlenbeckActionNoise as OUNoise
from model.ddpg import DDPG
from env.multirotor import Multirotor
from utils import save_results, make_dir, plot_rewards, save_args

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path


def get_args():
    """
    Hyper parameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='DDPG', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='UE4 and Airsim', type=str, help="name of environment")
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    parser.add_argument('--n_state', default=3 + 3 + 3 + 1 + 37, type=int, help="numbers of state space")
    parser.add_argument('--n_action', default=3, type=int, help="numbers of state action")
    parser.add_argument('--train_eps', default=8000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=300, type=int, help="episodes of testing")
    parser.add_argument('--max_step', default=2000, type=int, help="max step for getting target")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-3, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=20000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--soft_tau', default=1e-2, type=float)
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                 '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                '/' + curr_time + '/models/')  # path to save models
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # check GPU
    return args


def set_seed(seed):
    """
    全局生效
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(cfg, client, agent):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    ou_noise = OUNoise(mu=np.zeros(cfg.n_action))  # noise of action
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    writer = SummaryWriter('./train_image')
    for i_ep in range(cfg.train_eps):
        env = Multirotor(client)
        state = env.get_state()
        ou_noise.reset()
        ep_reward = 0
        for i_step in range(cfg.max_step):
            env.current_set(client)
            action = agent.choose_action(state)
            action = action + ou_noise(t=i_step)  # 动作加噪音
            # 加速度范围 ax[-1,1] ay[-1,1] az[-1,1] 速度大致范围 vx[-11,11] vy[-11,11] vz[-8,8]
            action = np.clip(action, -0.5, 0.5)  # 裁剪
            next_state, reward, done = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if done:
                break
        if (i_ep + 1) % 10 == 0:
            print(f'Env:{i_ep + 1}/{cfg.train_eps}, Reward:{ep_reward:.2f}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        writer.add_scalars(main_tag='train',
                           tag_scalar_dict={
                               'reward': ep_reward,
                               'ma_reward': ma_rewards[-1]
                           },
                           global_step=i_ep)
    writer.close()
    print('Finish training!')
    return rewards, ma_rewards


if __name__ == '__main__':
    cfg = get_args()
    set_seed(cfg.seed)
    client = airsim.MultirotorClient()  # connect to the AirSim simulator
    agent = DDPG(cfg=cfg)
    rewards, ma_rewards = train(cfg, client, agent)

    make_dir(cfg.result_path, cfg.model_path)
    save_args(cfg)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, cfg, tag="train")  # 画出结果
