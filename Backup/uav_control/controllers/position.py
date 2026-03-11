"""
模块功能说明：
位置控制器 (Position Controller)
实现外环位置PID控制，输出期望速度。

重要参数：
- config['kp']: 比例增益 [x, y, z]

使用示例：
pos_ctrl = PositionController(config)
target_vel = pos_ctrl.update(target_pos, current_pos)
"""

import numpy as np
from .base_controller import BaseController

class PositionController(BaseController):
    def __init__(self, config):
        super().__init__(config)
        self.kp = np.array(config['kp'])
        self.output_limits = config.get('output_limits', [None, None])

    def update(self, setpoint, measurement):
        # 计算误差
        error = np.array(setpoint) - np.array(measurement)
        
        # P控制
        output = self.kp * error
        
        # 限幅
        min_val, max_val = self.output_limits
        if min_val is not None and max_val is not None:
            output = np.clip(output, min_val, max_val)
            
        return output

    def reset(self):
        pass
