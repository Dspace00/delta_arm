"""
控制器基类模块
定义所有控制器的公共接口
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class ControllerBase(ABC):
    """控制器基类"""
    
    def __init__(self, name: str = "Unnamed Controller"):
        self.name = name
        self.enabled = True
        
    @abstractmethod
    def update(self, setpoint: Any, current_value: Any) -> Any:
        """更新控制器状态并返回控制输出"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置控制器状态"""
        pass
    
    def enable(self):
        """启用控制器"""
        self.enabled = True
        
    def disable(self):
        """禁用控制器"""
        self.enabled = False
        
    def __str__(self):
        return f"{self.name} (enabled: {self.enabled})"


class PIDControllerBase(ControllerBase):
    """PID控制器基类"""
    
    def __init__(self, kp, ki, kd, dt, output_limits=(-float('inf'), float('inf')), name="PID Controller"):
        super().__init__(name)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        self.integral = 0.0
        self.previous_error = 0.0
        
    def _limit_output(self, output):
        """限制输出范围"""
        if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
            return np.clip(output, self.output_limits[0], self.output_limits[1])
        return output