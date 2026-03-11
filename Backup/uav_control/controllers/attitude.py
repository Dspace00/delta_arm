"""
模块功能说明：
姿态控制器 (Attitude Controller)
实现内环姿态PID控制，输出期望角速度。

重要参数：
- config['kp']: 比例增益

使用示例：
att_ctrl = AttitudeController(config)
target_rate = att_ctrl.update(target_quat, current_quat)
"""

import numpy as np
import quaternion
from .base_controller import BaseController

class AttitudeController(BaseController):
    def __init__(self, config):
        super().__init__(config)
        self.kp = np.array(config['kp'])

    def update(self, setpoint, measurement):
        """
        setpoint: target quaternion
        measurement: current quaternion
        """
        # Ensure inputs are quaternion objects
        if not isinstance(setpoint, np.quaternion):
            setpoint = np.quaternion(*setpoint) if len(setpoint)==4 else setpoint
        if not isinstance(measurement, np.quaternion):
            measurement = np.quaternion(*measurement) if len(measurement)==4 else measurement

        # Error quaternion: q_err = q_target * q_curr_conj
        q_err = setpoint * measurement.conjugate()
        
        # Extract vector part (x, y, z)
        # q_err = [w, x, y, z]
        # If w < 0, negate to ensure shortest path
        if q_err.w < 0:
            q_err = -q_err
            
        error_vec = np.array([q_err.x, q_err.y, q_err.z])
        
        # Control Law: rate_cmd = 2/tau * sign(q0) * q_vec ? 
        # Simple P controller: rate_cmd = Kp * error_vec
        # (Assuming small angles, error_vec ~ theta/2)
        
        output = self.kp * error_vec
        
        # Note: Previous code used Kp * error directly.
        
        return output

    def reset(self):
        pass
