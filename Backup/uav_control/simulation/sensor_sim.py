"""
模块功能说明：
定义传感器仿真接口，提供抽象基类和MuJoCo实现。

重要参数：
- model: MuJoCo模型
- data: MuJoCo数据

使用示例：
sensor = SimSensor(model, data)
pos, vel, quat, rate = sensor.get_state()
"""

import numpy as np
import mujoco
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R
import quaternion

class SensorBase(ABC):
    @abstractmethod
    def get_state(self):
        """
        返回: (position, velocity, attitude_quat, angular_rate)
        position: [x, y, z] (World)
        velocity: [vx, vy, vz] (World)
        attitude_quat: [w, x, y, z] (World->Body or Body->World? Usually Body attitude in World)
        angular_rate: [wx, wy, wz] (Body frame)
        """
        pass

class SimSensor(SensorBase):
    def __init__(self, model, data, body_name="UAV_body"):
        self.model = model
        self.data = data
        self.body_name = body_name
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id == -1:
            # fallback to 0 if not found, or raise
            self.body_id = 1 # Root body often 1? Or 0 is world. 
            # In previous code it defaulted to finding it.
            pass

    def get_state(self):
        # Position (World)
        position = self.data.body(self.body_id).xpos.copy()
        
        # Attitude (Quat [w,x,y,z])
        # MuJoCo xquat is [w, x, y, z]
        mj_quat = self.data.body(self.body_id).xquat.copy()
        # Ensure it is normalized? MuJoCo keeps it normalized usually.
        attitude = np.quaternion(mj_quat[0], mj_quat[1], mj_quat[2], mj_quat[3])
        
        # Velocity (World) - cvel is 6D [ang, lin] in world space? No, cvel is com velocity in world.
        # Check previous code: velocity = self.data.body(body_id).cvel[3:6].copy()
        # Note: cvel is [rot_vel, lin_vel]
        velocity = self.data.body(self.body_id).cvel[3:6].copy()
        
        # Angular Rate (Body)
        # Previous code: 
        # angle_rate_world = self.data.body(body_id).cvel[0:3].copy()
        # body_att_mat = self.data.body(body_id).xmat.copy().reshape(3,3)
        # angle_rate = body_att_mat.T @ angle_rate_world
        
        angle_rate_world = self.data.body(self.body_id).cvel[0:3].copy()
        body_att_mat = self.data.body(self.body_id).xmat.copy().reshape(3,3)
        angle_rate = body_att_mat.T @ angle_rate_world
        
        return position, velocity, attitude, angle_rate
