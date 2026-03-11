"""
无人机配置
"""
from .base_config import ConfigBase
from typing import Dict, Any


class MotorConfig(ConfigBase):
    """电机配置"""
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "count": 4,
            "max_thrust": 6.5,  # N
            "max_torque": 0.15,  # Nm
            "max_speed": 22,  # krpm
            "thrust_coefficient": 0.01343,  # Ct (N/krpm^2)
            "torque_coefficient": 3.099e-4,  # Cd (Nm/krpm^2)
            "arm_length": 0.18,  # m
            "layout": "x",  # 'x' 或 '+' 型
            "motor_order": [0, 1, 2, 3],  # 电机顺序
            "rotation_direction": [1, -1, 1, -1]  # 旋转方向 (1: 逆时针, -1: 顺时针)
        }


class DroneConfig(ConfigBase):
    """无人机配置"""
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "name": "Quadrotor",
            "mass": 1.27,  # kg
            "inertia": {
                "Ixx": 0.022,
                "Iyy": 0.022,
                "Izz": 0.036
            },
            "dimensions": {
                "arm_length": 0.18,  # m
                "height": 0.05,  # m
                "propeller_diameter": 0.1  # m
            },
            "motors": MotorConfig().to_dict(),
            "gravity": 9.81,
            "drag_coefficient": 0.1,
            "max_tilt_angle": 30.0,  # 最大倾斜角度 (度)
            "body_name": "UAV_body"  # MuJoCo中的刚体名称
        }
    
    def validate(self) -> bool:
        """验证无人机配置"""
        if self.mass <= 0:
            raise ValueError("Mass must be positive")
        if self.dimensions["arm_length"] <= 0:
            raise ValueError("Arm length must be positive")
        if self.motors["count"] < 2:
            raise ValueError("At least 2 motors required")
        return True