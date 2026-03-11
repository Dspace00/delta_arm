"""
无人机模型定义
封装无人机的状态和物理参数
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import quaternion


@dataclass
class DroneState:
    """无人机状态数据类"""
    position: np.ndarray        # [x, y, z] 位置 (m)
    velocity: np.ndarray        # [vx, vy, vz] 线速度 (m/s)
    attitude: np.quaternion     # 四元数姿态 [w, x, y, z]
    angular_velocity: np.ndarray # [wx, wy, wz] 角速度 (rad/s)
    timestamp: float            # 时间戳 (s)
    
    def __post_init__(self):
        """初始化后验证数据"""
        if len(self.position) != 3:
            raise ValueError("position must be a 3-element array")
        if len(self.velocity) != 3:
            raise ValueError("velocity must be a 3-element array")
        if len(self.angular_velocity) != 3:
            raise ValueError("angular_velocity must be a 3-element array")
    
    def to_dict(self):
        """转换为字典"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'attitude': self.attitude,
            'angular_velocity': self.angular_velocity.copy(),
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建实例"""
        return cls(
            position=data['position'],
            velocity=data['velocity'],
            attitude=data['attitude'],
            angular_velocity=data['angular_velocity'],
            timestamp=data['timestamp']
        )


class DroneModel:
    """无人机模型类"""
    
    def __init__(self, mass: float = 1.27, arm_length: float = 0.18):
        """
        初始化无人机模型
        
        Args:
            mass: 无人机质量 (kg)
            arm_length: 机臂长度 (m)
        """
        self.mass = mass
        self.arm_length = arm_length
        self.gravity = 9.81
        
        # 物理参数
        self.max_thrust_per_motor = 6.5  # N
        self.max_torque_per_motor = 0.15  # Nm
        
        # 当前状态
        self.state: Optional[DroneState] = None
        self.history = []
        
    def update_state(self, new_state: DroneState):
        """更新无人机状态"""
        self.state = new_state
        self.history.append(new_state)
        
    def get_current_state(self) -> DroneState:
        """获取当前状态"""
        if self.state is None:
            raise ValueError("Drone state not initialized")
        return self.state
    
    def calculate_thrust_to_hover(self) -> float:
        """计算悬停所需的总推力"""
        return self.mass * self.gravity
    
    def __str__(self):
        if self.state is None:
            return f"DroneModel(mass={self.mass}kg, state=NotInitialized)"
        pos = self.state.position
        return f"DroneModel(mass={self.mass}kg, position=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}])"