import airsim
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from model.ddpg import DDPG
from env.multirotor import Multirotor
from utils import save_results, plot_rewards
from train import get_args, set_seed


def test(cfg, client, agent):
    print('Start testing')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    writer = SummaryWriter('./test_image')
    success = 0
    for i_ep in range(cfg.test_eps):
        env = Multirotor(client)
        state = env.get_state()
        ep_reward = 0
        finish_step = 0
        final_distance = state[3]
        for i_step in range(cfg.max_step):
            finish_step = finish_step + 1
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            state = next_state
            print('\rEpisode: {}\tStep: {}\tReward: {:.2f}\tDistance: {}'.format(i_ep + 1, i_step + 1, ep_reward,
                                                                                 state[3]), end="")
            final_distance = state[3]
            if done:
                break
        print('\rEpisode: {}\tFinish step: {}\tReward: {:.2f}\tFinal distance: {}'.format(i_ep + 1, finish_step,
                                                                                          ep_reward, final_distance))
        if final_distance <= 10.0:
            success += 1
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        writer.add_scalars(main_tag='test',
                           tag_scalar_dict={
                               'reward': ep_reward,
                               'ma_reward': ma_rewards[i_ep]
                           },
                           global_step=i_ep)
    print('Finish testing!')
    print('Average Reward: {}\tSuccess Rate: {}'.format(np.mean(rewards), success / cfg.test_eps))
    writer.close()

    return rewards, ma_rewards


if __name__ == '__main__':
    cfg = get_args()
    set_seed(cfg.seed)
    client = airsim.MultirotorClient()  # connect to the AirSim simulator
    agent = DDPG(cfg=cfg)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = test(cfg, client, agent)
    save_results(rewards, ma_rewards, tag='test', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, cfg, tag="test")  # 画出结果
