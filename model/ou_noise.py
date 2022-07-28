import numpy as np
import torch


class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein–Uhlenbeck噪声
    参考：https://zhuanlan.zhihu.com/p/96720878
    相比于独立噪声（高斯噪音），OU噪声适合于惯性系统，尤其是时间离散化粒度较小的情况，
    在时间离散化粒度不小的情况下（如0.1s），独立噪音也有不错的效果
    mu, theta, sigma是OU噪声的参数，均为正值
    mu为均值，要在其上施加噪声，在类中传入的mu需要时一个“与动作相关的向量”，如np.zeros(action_dim)
    sigma是维纳过程的参数，决定了噪声放大的倍数
    theta值越大，向均值靠近的速度越快，由噪声回归均值的时间更短
    """

    def __init__(self, mu, theta=0.15, max_sigma=0.3, min_sigma=0.1, dt=1e-2, x0=None, decay_period=100000):
        self.x_prev = None  # 没有施加噪声的原值（不是action值，原action加上这个值才是OU处理过的action）
        self.mu = mu  # OU噪声的参数
        self.theta = theta  # OU噪声的参数
        self.sigma = max_sigma  # OU噪声的参数
        self.dt = dt
        self.x0 = x0
        self.reset()

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    # 直接调用得到噪音，该噪音要加在action上，加完后在使用np.clip()对action进行裁剪，限制其范围
    def __call__(self, t=0):
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)  # sigma会逐渐衰减
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    # def sample(self):
    #     dx = self.theta * (self.mu - self.X)
    #     dx = dx + self.sigma * np.random.randn(self.n_actions)
    #     self.X = self.X + dx
    #     return self.X
    #
    # def get_action(self, action, t=0):
    #     ou_x = self.sample()  # 经过噪声处理的值
    #     self.sigma = self.max_sigma -
    #     (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)  # sigma会逐渐衰减
    #     return np.clip(action + ou_x, self.low, self.high)  # 动作加上噪声后进行剪切


if __name__ == '__main__':
    action = np.array([.25, .1, .7])
    n_action = action.shape[0]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_action))
    print(ou_noise())
    ou_noise.reset()
    for i in range(100):
        print(ou_noise(i))
    # action = action + ou_noise(t=100)  # 动作加噪音
    # print(action)
    # action = np.clip(action, -.5, .5)  # 裁剪
    # print(action[0], action[1], action[2])
