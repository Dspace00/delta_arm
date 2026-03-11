"""
仿真配置
"""
from .base_config import ConfigBase
from typing import Dict, Any


class VisualizationConfig(ConfigBase):
    """可视化配置"""
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "fps": 60,
            "camera": {
                "distance": 5.0,
                "azimuth": 45.0,
                "elevation": -30.0,
                "trackbodyid": -1
            },
            "show_target": True,
            "show_trajectory": True,
            "show_forces": False
        }


class LoggingConfig(ConfigBase):
    """日志配置"""
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "directory": "./logs",
            "filename_prefix": "simulation",
            "save_frequency": 10,  # 每10步保存一次
            "save_formats": ["csv", "json"],  # 'csv', 'json', 'hdf5'
            "log_variables": [
                "time",
                "position", "velocity", "attitude",
                "target_position", "target_velocity", "target_attitude",
                "motor_thrusts", "control_outputs"
            ],
            "max_file_size_mb": 100
        }


class SimulationConfig(ConfigBase):
    """仿真配置"""
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "name": "Drone Simulation",
            "model_path": "meshes/Delta.xml",
            "duration": 10.0,  # 秒
            "timestep": 0.0001,  # 秒
            "real_time_factor": 1.0,  # 实时因子
            "enable_physics": True,
            "enable_wind": False,
            "wind_speed": [0.0, 0.0, 0.0],  # m/s
            "noise": {
                "position": 0.001,  # 位置噪声 (m)
                "velocity": 0.01,   # 速度噪声 (m/s)
                "attitude": 0.001,  # 姿态噪声 (rad)
                "gyro": 0.01        # 陀螺仪噪声 (rad/s)
            },
            "visualization": VisualizationConfig().to_dict(),
            "logging": LoggingConfig().to_dict(),
            "initial_conditions": {
                "position": [0.0, 0.0, 1.0],
                "velocity": [0.0, 0.0, 0.0],
                "attitude": [1.0, 0.0, 0.0, 0.0],  # [w, x, y, z]
                "angular_velocity": [0.0, 0.0, 0.0]
            }
        }