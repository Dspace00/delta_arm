"""
速度控制器
"""
import numpy as np
from typing import Union, List, Optional
from .pid_controller import PIDController, PIDParams


class VelocityController:
    """速度控制器（PID控制器）"""
    
    def __init__(self, config, mass: float = 1.27, name: str = "Velocity Controller"):
        """
        初始化速度控制器
        
        Args:
            config: 速度控制器配置
            mass: 无人机质量 (kg)
            name: 控制器名称
        """
        self.name = name
        self.config = config
        self.mass = mass
        self.gravity = 9.81
        
        # 创建PID控制器
        pid_config = config.get('pid', {})
        params = PIDParams(
            kp=pid_config.get('kp', [1.0, 1.0, 1.0]),
            ki=pid_config.get('ki', [0.0, 0.0, 0.0]),
            kd=pid_config.get('kd', [0.0, 0.0, 0.0]),
            dt=1.0 / config.get('frequency', 50.0),
            output_limit=pid_config.get('output_limit', [-10.0, 10.0])
        )
        
        self.pid = PIDController(params, name=f"{name}_PID")
        
        # 重力补偿
        self.gravity_compensation = np.array([0, 0, -self.gravity])
        
        # 状态变量
        self.previous_velocity = np.zeros(3)
        self.acceleration_estimate = np.zeros(3)
        
    def update(self, target_velocity: np.ndarray, current_velocity: np.ndarray, 
               dt: Optional[float] = None) -> np.ndarray:
        """
        更新速度控制器
        
        Args:
            target_velocity: 目标速度 [vx, vy, vz]
            current_velocity: 当前速度 [vx, vy, vz]
            dt: 时间步长（如果为None则使用控制器dt）
            
        Returns:
            加速度命令 [ax, ay, az] (m/s²)
        """
        if dt is None:
            dt = self.pid.params.dt
        
        # 计算加速度估计（用于微分项）
        if dt > 0:
            self.acceleration_estimate = (current_velocity - self.previous_velocity) / dt
            self.previous_velocity = current_velocity.copy()
        
        # 速度控制
        acceleration_pid = self.pid.update(target_velocity, current_velocity)
        
        # 重力补偿
        acceleration_command = acceleration_pid - self.gravity_compensation
        
        return acceleration_command
    
    def calculate_thrust_vector(self, acceleration_command: np.ndarray) -> np.ndarray:
        """
        计算推力向量
        
        Args:
            acceleration_command: 加速度命令 [ax, ay, az]
            
        Returns:
            推力向量 [Fx, Fy, Fz]
        """
        thrust_vector = self.mass * acceleration_command
        return thrust_vector
    
    def reset(self):
        """重置控制器"""
        self.pid.reset()
        self.previous_velocity = np.zeros(3)
        self.acceleration_estimate = np.zeros(3)
    
    def __str__(self):
        return f"VelocityController(name={self.name}, mass={self.mass}kg)"