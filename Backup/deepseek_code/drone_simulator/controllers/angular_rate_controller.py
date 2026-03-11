"""
角速度控制器
"""
import numpy as np
from typing import Union, List, Optional
from .pid_controller import PIDController, PIDParams


class AngularRateController:
    """角速度控制器（PID控制器）"""
    
    def __init__(self, config, inertia: np.ndarray = None, name: str = "Angular Rate Controller"):
        """
        初始化角速度控制器
        
        Args:
            config: 角速度控制器配置
            inertia: 转动惯量矩阵 [Ixx, Iyy, Izz]
            name: 控制器名称
        """
        self.name = name
        self.config = config
        
        # 转动惯量
        if inertia is None:
            self.inertia = np.array([0.022, 0.022, 0.036])  # 默认值
        else:
            self.inertia = np.asarray(inertia)
        
        # 创建PID控制器
        pid_config = config.get('pid', {})
        params = PIDParams(
            kp=pid_config.get('kp', [1.0, 1.0, 1.0]),
            ki=pid_config.get('ki', [0.0, 0.0, 0.0]),
            kd=pid_config.get('kd', [0.0, 0.0, 0.0]),
            dt=1.0 / config.get('frequency', 1000.0),
            output_limit=pid_config.get('output_limit', [-2.0, 2.0])
        )
        
        self.pid = PIDController(params, name=f"{name}_PID")
        
        # 状态变量
        self.previous_rate = np.zeros(3)
        self.angular_acceleration_estimate = np.zeros(3)
        
    def update(self, target_rate: np.ndarray, current_rate: np.ndarray,
               dt: Optional[float] = None) -> np.ndarray:
        """
        更新角速度控制器
        
        Args:
            target_rate: 目标角速度 [wx, wy, wz]
            current_rate: 当前角速度 [wx, wy, wz]
            dt: 时间步长（如果为None则使用控制器dt）
            
        Returns:
            扭矩命令 [Mx, My, Mz] (Nm)
        """
        if dt is None:
            dt = self.pid.params.dt
        
        # 计算角加速度估计
        if dt > 0:
            self.angular_acceleration_estimate = (current_rate - self.previous_rate) / dt
            self.previous_rate = current_rate.copy()
        
        # 角速度控制
        torque_pid = self.pid.update(target_rate, current_rate)
        
        # 考虑转动惯量（可选：用于前馈补偿）
        # torque_inertia = self.inertia * self.angular_acceleration_estimate
        
        # 总扭矩命令
        torque_command = torque_pid  # + torque_inertia（如果启用前馈）
        
        return torque_command
    
    def reset(self):
        """重置控制器"""
        self.pid.reset()
        self.previous_rate = np.zeros(3)
        self.angular_acceleration_estimate = np.zeros(3)
    
    def __str__(self):
        return f"AngularRateController(name={self.name}, inertia={self.inertia})"