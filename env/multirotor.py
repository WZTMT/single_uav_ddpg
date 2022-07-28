import airsim
import numpy as np
import math

from airsim import MultirotorClient
from math import *
from airsim import YawMode


def normal_func(x):
    y = -0.8 * (1 / (1 + math.exp(-x / 40)) - 0.5)
    return y


class Multirotor:
    def __init__(self, client: MultirotorClient):
        self.client = client

        client.reset()
        client.enableApiControl(True)  # 获取控制权
        client.armDisarm(True)  # 解锁py
        client.takeoffAsync().join()  # 起飞
        client.moveToZAsync(-1, 1).join()

        kinematic_state = self.client.simGetGroundTruthKinematics()

        # 无人机坐标
        self.ux = float(kinematic_state.position.x_val)
        self.uy = float(kinematic_state.position.y_val)
        self.uz = float(kinematic_state.position.z_val)

        # 无人机速度
        self.vx = float(kinematic_state.linear_velocity.x_val)
        self.vy = float(kinematic_state.linear_velocity.y_val)
        self.vz = float(kinematic_state.linear_velocity.z_val)

        self.bound_x = [-500, 500]
        self.bound_y = [-500, 500]
        self.bound_z = [-200, 0]
        self.target_x = [-350, -150]
        self.target_y = [250, 450]
        self.target_z = [-100, -50]
        self.d_safe = 15

        # 目标点坐标
        self.tx, self.ty, self.tz = self.generate_target()

    def generate_target(self):
        """
        生成目标点的位置
        seed为随机种子
        """
        # TODO 尝试修改随机种子
        tx = np.random.rand() * (self.target_x[1] - self.target_x[0]) + self.target_x[0]
        ty = np.random.rand() * (self.target_y[1] - self.target_y[0]) + self.target_y[0]
        tz = np.random.rand() * (self.target_z[1] - self.target_z[0]) + self.target_z[0]
        return tx, ty, tz

    '''
    获取无人机与目标点的连线与无人机第一视角方向（飞行方向）的夹角
    tx,ty,tz为目标点坐标
    '''

    def get_deflection_angle(self):
        # 连线向量
        ax = self.tx - self.ux
        ay = self.ty - self.uy
        az = self.tz - self.uz

        # 速度方向向量
        bx = self.vx
        by = self.vy
        bz = self.vz

        if bx == 0 and by == 0 and bz == 0:  # 若无人机停止飞行，则判定完全偏航，给予一个惩罚
            return 180

        model_a = pow(ax ** 2 + ay ** 2 + az ** 2, 0.5)
        model_b = pow(bx ** 2 + by ** 2 + bz ** 2, 0.5)

        cos_ab = (ax * bx + ay * by + az * bz) / (model_a * model_b)
        radius = acos(cos_ab)  # 计算结果为弧度制，范围（0， PI），越小越好
        angle = np.rad2deg(radius)

        return angle

    def get_distance(self):
        xa = self.tx - self.ux
        ya = self.ty - self.uy
        za = self.tz - self.uz
        return pow(xa ** 2 + ya ** 2 + za ** 2, 0.5)

    '''
    距离传感器返回的距离数据，水平，竖直各半个圆周，每30度一采样，
    共13个数据，顺序为S、Y(1-6)、P(1-6)
    '''
    def get_distance_sensors_data(self):
        yaw_axis = ['Y', 'P']
        pitch_axis = ['1', '2', '3', '4', '5', '6']
        data = []
        prefix = "Distance"
        data.append(self.client.getDistanceSensorData(distance_sensor_name=prefix + 'S').distance)
        for i in yaw_axis:
            for j in pitch_axis:
                dsn = prefix + i + j
                data.append(self.client.getDistanceSensorData(distance_sensor_name=dsn).distance)

        return data

    '''
    重置无人机并返回初始状态(numpy.array)
    '''

    def get_state(self):
        position = np.array([self.tx - self.ux, self.ty - self.uy, self.tz - self.uz])
        target = np.array([self.get_distance()])
        velocity = np.array([self.vx, self.vy, self.vz])
        angle = np.array([self.get_deflection_angle()])
        sensor_data = np.array(self.get_distance_sensors_data())

        state = np.append(position, target)
        state = np.append(state, velocity)
        state = np.append(state, angle)
        state = np.append(state, sensor_data)

        return state

    '''
    计算当前状态下的奖励、是否完成
    加速度的坐标为NED，z轴加速度为负则向上飞行
    三个加速度有统一的范围
    '''

    def step(self, action):
        done = self.if_done()
        arrive_reward = self.arrive_reward()
        yaw_reward = self.yaw_reward()
        min_sensor_reward = self.min_sensor_reward()
        num_sensor_reward = self.num_sensor_reward()
        collision_reward = 0
        step_reward = self.step_reward()
        distance_reward = self.distance_reward()
        cross_border_reward = self.cross_border_reward()

        # 碰撞完成与碰撞奖励一起做
        if self.client.simGetCollisionInfo().has_collided:
            done = True
            collision_reward = -50

        ax = action[0]
        ay = action[1]
        az = action[2]
        my_yaw_mode = YawMode()
        my_yaw_mode.is_rate = False
        my_yaw_mode.yaw_or_rate = 0
        # TODO 尝试改为速度控制
        self.client.moveByVelocityAsync(vx=self.vx + ax,
                                        vy=self.vy + ay,
                                        vz=self.vz + az,
                                        duration=0.5,
                                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                                        yaw_mode=my_yaw_mode).join()
        reward = arrive_reward + yaw_reward + min_sensor_reward + num_sensor_reward + collision_reward + \
                 step_reward + distance_reward + cross_border_reward

        kinematic_state = self.client.simGetGroundTruthKinematics()
        self.ux = float(kinematic_state.position.x_val)
        self.uy = float(kinematic_state.position.y_val)
        self.uz = float(kinematic_state.position.z_val)
        self.vx = float(kinematic_state.linear_velocity.x_val)
        self.vy = float(kinematic_state.linear_velocity.y_val)
        self.vz = float(kinematic_state.linear_velocity.z_val)
        next_state = self.get_state()

        return next_state, reward, done

    def if_done(self):
        # 与目标点距离小于10米
        model_a = self.get_distance()
        if model_a <= 10.0:
            return True
        # 发生碰撞

        # 触及边界
        if self.ux < self.bound_x[0] or self.ux > self.bound_x[1] or \
                self.uy < self.bound_y[0] or self.uy > self.bound_y[1] or \
                self.uz < self.bound_z[0]:
            return True

        return False

    '''
    越界惩罚-20
    '''

    def cross_border_reward(self):
        if self.ux < self.bound_x[0] or self.ux > self.bound_x[1] or \
                self.uy < self.bound_y[0] or self.uy > self.bound_y[1] or \
                self.uz < self.bound_z[0]:
            return -20
        return 0

    '''
    与目标点距离惩罚(-0.4,0)
    '''

    def distance_reward(self):
        model_a = self.get_distance()
        return normal_func(model_a)

    '''
    抵达目标点奖励+50
    '''

    def arrive_reward(self):
        x = self.tx - self.ux
        y = self.ty - self.uy
        z = self.tz - self.uz
        model_a = pow(x ** 2 + y ** 2 + z ** 2, 0.5)
        if model_a <= 10.0:
            return 50
        else:
            return 0

    '''
    偏航惩罚(-0.1,0)
    '''

    def yaw_reward(self):
        yaw = self.get_deflection_angle()
        return -0.1 * (yaw / 180)

    '''
    最短激光雷达长度惩罚(-0.5,0)  
    '''

    def min_sensor_reward(self):
        sensor_data = self.get_distance_sensors_data()
        d_min = min(sensor_data)
        if d_min < self.d_safe:
            return -0.1 * (self.d_safe - d_min)
        else:
            return 0

    '''
    小于安全阈值的激光雷达条数惩罚(-0.3,0)
    '''

    def num_sensor_reward(self):
        sensor_data = self.get_distance_sensors_data()
        num = sum(i < self.d_safe for i in sensor_data)
        return -0.5 * (num / len(sensor_data))

    '''
    碰撞惩罚-50
    '''

    '''
    漫游惩罚-0.02
    '''

    def step_reward(self):
        if not self.if_done():
            return -0.02
        else:
            return 0


if __name__ == '__main__':
    normal_func = -0.8 * (1 / (1 + math.exp(-10 / 40)) - 0.5)
    print(normal_func)
