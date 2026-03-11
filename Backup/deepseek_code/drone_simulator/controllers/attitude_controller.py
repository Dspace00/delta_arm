"""
姿态控制器
"""
import numpy as np
import quaternion
from typing import Union, List, Optional, Tuple
from .pid_controller import PIDController, PIDParams
from scipy.spatial.transform import Rotation as R


class AttitudeController:
    """姿态控制器"""
    
    def __init__(self, config, name: str = "Attitude Controller"):
        """
        初始化姿态控制器
        
        Args:
            config: 姿态控制器配置
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
            dt=1.0 / config.get('frequency', 250.0),
            output_limit=pid_config.get('output_limit', [-5.0, 5.0])
        )
        
        self.pid = PIDController(params, name=f"{name}_PID")
        
        # 偏航角控制参数
        self.yaw_angle = 0.0  # 目标偏航角（弧度）
        self.yaw_weight = 0.1  # 偏航控制权重
        
    def update(self, target_attitude: Union[np.quaternion, np.ndarray],
               current_attitude: Union[np.quaternion, np.ndarray]) -> np.ndarray:
        """
        更新姿态控制器
        
        Args:
            target_attitude: 目标姿态（四元数）
            current_attitude: 当前姿态（四元数）
            
        Returns:
            角速度命令 [wx, wy, wz] (rad/s)
        """
        # 转换为四元数对象
        if isinstance(target_attitude, np.ndarray):
            target_quat = quaternion.from_float_array(target_attitude)
        else:
            target_quat = target_attitude
            
        if isinstance(current_attitude, np.ndarray):
            current_quat = quaternion.from_float_array(current_attitude)
        else:
            current_quat = current_attitude
        
        # 计算姿态误差（四元数差）
        error_quat = target_quat * current_quat.conjugate()
        
        # 确保表示最短路径
        if error_quat.w < 0:
            error_quat = -error_quat
        
        # 将四元数误差转换为轴角表示
        angle = 2 * np.arccos(np.clip(error_quat.w, -1.0, 1.0))
        axis = np.array([error_quat.x, error_quat.y, error_quat.z])
        
        # 避免除以零
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            error_vector = axis * angle
        else:
            error_vector = np.zeros(3)
        
        # 姿态控制（P控制器）
        angular_velocity_command = self.pid.update(np.zeros(3), -error_vector)
        
        return angular_velocity_command
    
    def calculate_desired_attitude(self, thrust_vector: np.ndarray, 
                                   target_yaw: float = 0.0) -> np.quaternion:
        """
        根据推力向量和偏航角计算期望姿态
        
        Args:
            thrust_vector: 推力向量 [Fx, Fy, Fz]
            target_yaw: 目标偏航角（弧度）
            
        Returns:
            期望姿态（四元数）
        """
        # 归一化推力向量得到机体Z轴
        thrust_norm = np.linalg.norm(thrust_vector)
        if thrust_norm < 1e-6:
            return quaternion.from_float_array([1, 0, 0, 0])
        
        z_body = thrust_vector / thrust_norm
        
        # 创建偏航向量（世界坐标系中的偏航方向）
        yaw_vector = np.array([-np.sin(target_yaw), np.cos(target_yaw), 0])
        
        # 计算X轴（与偏航向量和Z轴垂直）
        x_body = np.cross(yaw_vector, z_body)
        x_norm = np.linalg.norm(x_body)
        
        if x_norm < 1e-6:
            # 如果偏航向量与Z轴平行，选择默认X轴
            if abs(z_body[2]) < 0.9:
                x_body = np.array([z_body[1], -z_body[0], 0])
            else:
                x_body = np.array([0, z_body[2], -z_body[1]])
            x_norm = np.linalg.norm(x_body)
        
        x_body = x_body / x_norm
        
        # 计算Y轴
        y_body = np.cross(z_body, x_body)
        y_body = y_body / np.linalg.norm(y_body)
        
        # 构建旋转矩阵
        rotation_matrix = np.column_stack((x_body, y_body, z_body))
        
        # 转换为四元数
        rotation = R.from_matrix(rotation_matrix)
        quat = quaternion.from_float_array(rotation.as_quat())
        
        return quat
    
    def reset(self):
        """重置控制器"""
        self.pid.reset()
    
    def __str__(self):
        return f"AttitudeController(name={self.name})"