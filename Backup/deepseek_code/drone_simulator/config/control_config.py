"""
控制器配置
"""
from .base_config import ConfigBase
from typing import Dict, Any, List


class PIDConfig(ConfigBase):
    """PID控制器配置"""
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "kp": [1.0, 1.0, 1.0],  # 比例增益
            "ki": [0.0, 0.0, 0.0],  # 积分增益
            "kd": [0.0, 0.0, 0.0],  # 微分增益
            "integral_limit": 1.0,  # 积分限幅
            "output_limit": [-10.0, 10.0],  # 输出限幅
            "anti_windup": "clamping",  # 抗饱和方法: 'clamping', 'back-calculation'
            "dt": 0.02  # 采样时间 (秒)
        }


class ControllerConfig(ConfigBase):
    """控制器配置"""
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "position": {
                "frequency": 50,  # Hz
                "pid": {
                    "kp": [8.0, 8.0, 12.0],
                    "ki": [0.0, 0.0, 0.0],
                    "kd": [0.0, 0.0, 0.0],
                    "output_limit": [-5.0, 5.0]
                }
            },
            "velocity": {
                "frequency": 50,  # Hz
                "pid": {
                    "kp": [4.0, 4.0, 6.0],
                    "ki": [0.0, 0.0, 0.0],
                    "kd": [0.05, 0.05, 0.05],
                    "output_limit": [-10.0, 10.0]
                }
            },
            "attitude": {
                "frequency": 250,  # Hz
                "pid": {
                    "kp": [9.0, 9.0, 12.0],
                    "ki": [0.0, 0.0, 0.0],
                    "kd": [0.0, 0.0, 0.0],
                    "output_limit": [-5.0, 5.0]
                }
            },
            "angle_rate": {
                "frequency": 1000,  # Hz
                "pid": {
                    "kp": [3.0, 3.0, 3.0],
                    "ki": [100.0, 100.0, 100.0],
                    "kd": [0.00005, 0.00005, 0.00005],
                    "output_limit": [-2.0, 2.0]
                }
            },
            "mixer_type": "standard",  # 'standard', 'pseudo_inverse'
            "control_mode": "attitude",  # 'attitude', 'rate', 'velocity', 'position'
            "enable_anti_windup": True,
            "enable_feedforward": False
        }