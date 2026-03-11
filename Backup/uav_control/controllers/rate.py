"""
模块功能说明：
角速率控制器 (Rate Controller)
实现底层角速率PID控制，输出力矩。

重要参数：
- config['kp/ki/kd']: PID增益

使用示例：
rate_ctrl = RateController(config)
torques = rate_ctrl.update(target_rate, current_rate)
"""

import numpy as np
from .base_controller import BaseController

class RateController(BaseController):
    def __init__(self, config):
        super().__init__(config)
        self.kp = np.array(config['kp'])
        self.ki = np.array(config.get('ki', [0,0,0]))
        self.kd = np.array(config.get('kd', [0,0,0]))
        self.dt = 1.0 / 1000.0
        
        self.integral = np.zeros(3)
        self.last_measurement = np.zeros(3)

    def set_dt(self, dt):
        self.dt = dt

    def update(self, setpoint, measurement):
        error = np.array(setpoint) - np.array(measurement)
        
        p_term = self.kp * error
        
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        
        # Derivative on measurement
        v_dot = (measurement - self.last_measurement) / self.dt
        d_term = self.kd * (0 - v_dot)
        
        self.last_measurement = measurement
        
        output = p_term + i_term + d_term
        return output

    def reset(self):
        self.integral = np.zeros(3)
        self.last_measurement = np.zeros(3)
