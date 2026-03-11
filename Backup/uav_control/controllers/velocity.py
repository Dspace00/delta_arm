"""
模块功能说明：
速度控制器 (Velocity Controller)
实现中环速度PID控制，输出期望姿态(Quaternion)和总推力(Thrust)。

重要参数：
- config['kp/ki/kd']: PID增益
- mass: 无人机质量
- gravity: 重力加速度

注意事项：
- 输出包含两个部分：姿态(Quaternion)用于姿态控制，推力(float)用于混控。
- 包含从加速度向量到目标姿态的转换逻辑。
"""

import numpy as np
import quaternion
from .base_controller import BaseController

class VelocityController(BaseController):
    def __init__(self, config, mass=1.27, gravity=9.81):
        super().__init__(config)
        self.kp = np.array(config['kp'])
        self.ki = np.array(config.get('ki', [0,0,0]))
        self.kd = np.array(config.get('kd', [0,0,0]))
        self.dt = 1.0 / 50.0 # Default assumption, should pass in? or store time
        
        self.mass = mass
        self.gravity = gravity
        
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.last_measurement = np.zeros(3)

    def set_dt(self, dt):
        self.dt = dt

    def update(self, setpoint, measurement):
        # 1. PID Calculation to get Desired Acceleration (in World Frame)
        error = np.array(setpoint) - np.array(measurement)
        
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        
        # Derivative (on measurement to avoid derivative kick)
        # Or on error? Previous code: (current - last_val) / dt -> v_dot
        # derivative_term = self.kd * (0 - v_dot)
        v_dot = (measurement - self.last_measurement) / self.dt
        d_term = self.kd * (0 - v_dot)
        
        self.last_measurement = measurement
        self.prev_error = error
        
        acc_cmd = p_term + i_term + d_term
        
        # 2. Gravity Compensation
        # Target Force vector = Mass * (Acc_cmd - Gravity_vec)
        # Note: Gravity acts downwards (-z). To counteract, we need upward force (+z).
        # Eq: F_total / m + g = a_cmd  => F_total = m * (a_cmd - g)
        gravity_vec = np.array([0, 0, -self.gravity])
        total_force_vec = self.mass * (acc_cmd - gravity_vec)
        
        # 3. Convert Force Vector to Attitude and Thrust Magnitude
        thrust_magnitude = np.linalg.norm(total_force_vec)
        
        # Desired Z-axis (body up) should align with Force Vector
        z_b_des = total_force_vec / (thrust_magnitude)
        
        # Desired Yaw (assume 0 for now, or pass as extra arg)
        yaw_des = 0.0
        # Project Body-Y axis to XY plane? 
        # Standard approach:
        # x_c = [cos(yaw), sin(yaw), 0]
        # y_b_des = (z_b_des x x_c) / norm
        # x_b_des = y_b_des x z_b_des
        
        # Previous code approach:
        # yaw_axis = [-sin(yaw), cos(yaw), 0]  (Body Y projected?)
        # x_axis = cross(yaw_axis, z_axis) ...
        # Let's stick to a standard safe conversion
        
        yaw_vec = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        
        # y_b = z_b x x_c (if z_b is up, x_c is forward) -> y_b is left
        y_b_des = np.cross(z_b_des, yaw_vec)
        norm_y = np.linalg.norm(y_b_des)
        
        if norm_y < 1e-6:
            # Singularity (z_b aligned with x_c), unlikely for hovering
            y_b_des = np.array([0, 1, 0])
        else:
            y_b_des /= norm_y
            
        x_b_des = np.cross(y_b_des, z_b_des)
        
        # Rotation Matrix [x_b, y_b, z_b]
        rot_mat = np.column_stack((x_b_des, y_b_des, z_b_des))
        
        # Convert to Quaternion
        target_attitude = quaternion.from_rotation_matrix(rot_mat)
        
        return target_attitude, total_force_vec

    def reset(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.last_measurement = np.zeros(3)
