"""
位置控制器
"""
import numpy as np
import quaternion
from typing import Union, List, Optional
from .pid_controller import PIDController, PIDParams


class PositionController:
    """位置控制器（P控制器）"""
    
    def __init__(self, config, name: str = "Position Controller"):
        """
        初始化位置控制器
        
        Args:
            config: 位置控制器配置
            name: 控制器名称
        """
        self.name = name
        self.config = config
        
        # 创建PID控制器
        pid_config = config.get('pid', {})
        params = PIDParams(
            kp=pid_config.get('kp', [1.0, 1.0, 1.0]),
            ki=pid_config.get('ki', [0.0, 0.0, 0.0]),
            kd=pid_config.get('kd', [0.0, 0.0, 0.0]),
            dt=1.0 / config.get('frequency', 50.0),
            output_limit=pid_config.get('output_limit', [-5.0, 5.0])
        )
        
        self.pid = PIDController(params, name=f"{name}_PID")
        
        # 控制模式
        self.control_mode = 'position'  # 'position' 或 'velocity'
        
    def update(self, target_position: np.ndarray, current_position: np.ndarray) -> np.ndarray:
        """
        更新位置控制器
        
        Args:
            target_position: 目标位置 [x, y, z]
            current_position: 当前位置 [x, y, z]
            
        Returns:
            目标速度命令 [vx, vy, vz]
        """
        # 位置误差
        position_error = target_position - current_position
        
        # 位置控制（P控制器）
        velocity_command = self.pid.update(target_position, current_position)
        
        return velocity_command
    
    def reset(self):
        """重置控制器"""
        self.pid.reset()
    
    def set_target(self, position: np.ndarray):
        """设置目标位置"""
        self.target_position = position
    
    def __str__(self):
        return f"PositionController(name={self.name}, mode={self.control_mode})"