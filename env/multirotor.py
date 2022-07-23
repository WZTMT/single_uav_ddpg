from airsim import MultirotorClient
from math import *
import numpy as np


class Multirotor:
    def __init__(self, client: MultirotorClient):
        self.client = client
        kinematic_state = client.simGetGroundTruthKinematics()
        position = kinematic_state.position
        velocity = kinematic_state.linear_velocity

        # 无人机坐标
        self.ux = float(position.x_val)
        self.uy = float(position.y_val)
        self.uz = float(position.z_val)

        # 无人机速度
        self.vx = float(velocity.x_val)
        self.vy = float(velocity.y_val)
        self.vz = float(velocity.z_val)

    '''
    获取无人机与目标点的连线与无人机第一视角方向（飞行方向）的夹角
    tx,ty,tz为目标点坐标
    '''
    def get_deflection_angle(self, tx: float, ty: float, tz: float):
        # 连线向量
        ax = tx - self.ux
        ay = ty - self.uy
        az = tz - self.uz

        # 速度方向向量
        bx = self.vx
        by = self.vy
        bz = self.vz

        model_a = pow(ax ** 2 + ay ** 2 + az ** 2, 0.5)
        model_b = pow(bx ** 2 + by ** 2 + bz ** 2, 0.5)
        cos_ab = (ax * bx + ay * by + az * bz)/(model_a * model_b)
        radius = acos(cos_ab)  # 计算结果为弧度制，范围（0， PI），越小越好
        angle = np.rad2deg(radius)

        return angle

    '''
    距离传感器返回的距离数据，覆盖无人机的正面半个球体，每30度一采样，
    共37个数据，顺序为U、D、A(1-5)、B(1-5)、...、G(1-5)
    '''
    def get_distance_sensors_data(self):
        yaw_axis = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        pitch_axis = ['1', '2', '3', '4', '5']
        data = []
        prefix = "Distance"
        data.append(self.client.getDistanceSensorData(distance_sensor_name=prefix+'U').distance)
        data.append(self.client.getDistanceSensorData(distance_sensor_name=prefix+'D').distance)
        for i in yaw_axis:
            for j in pitch_axis:
                dsn = prefix + i + j
                data.append(self.client.getDistanceSensorData(distance_sensor_name=dsn).distance)

        return data
