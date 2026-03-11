"""
混控器（从原代码迁移）
"""
import numpy as np
from typing import List, Tuple, Optional


class Mixer:
    """动力分配混控器"""
    
    def __init__(self, config: Optional[dict] = None):
        """
        初始化混控器
        
        Args:
            config: 混控器配置
        """
        if config is None:
            config = {}
        
        # 物理参数
        self.Ct = config.get('thrust_coefficient', 0.01343)       # 电机推力系数 (N/krpm^2)
        self.Cd = config.get('torque_coefficient', 3.099e-4)      # 电机反扭系数 (Nm/krpm^2)
        self.L = config.get('arm_length', 0.18)                   # 电机力臂长度 (m)
        self.max_thrust = config.get('max_thrust', 6.5)           # 单个电机最大推力 (N)
        self.max_torque = config.get('max_torque', 0.15)          # 单个电机最大扭矩 (Nm)
        self.max_speed = config.get('max_speed', 22)              # 电机最大转速 (krpm)
        
        # 动力分配矩阵（X型配置）
        self.mat = np.array([
            [self.Ct, self.Ct, self.Ct, self.Ct],                   # 总推力
            [self.Ct*self.L, -self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L],  # 滚转力矩
            [-self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L, self.Ct*self.L],  # 俯仰力矩
            [-self.Cd, self.Cd, -self.Cd, self.Cd]                  # 偏航力矩
        ])
        
        # 动力分配逆矩阵
        self.inv_mat = np.linalg.inv(self.mat)
        
        # 电机布局
        self.motor_layout = config.get('layout', 'x')  # 'x' 或 '+'
        self.motor_count = config.get('count', 4)
    
    def allocate(self, thrust: float, torque: np.ndarray) -> np.ndarray:
        """
        动力分配
        
        Args:
            thrust: 总推力 (N)
            torque: 三轴扭矩 [Mx, My, Mz] (Nm)
            
        Returns:
            电机转速平方 [w1², w2², w3², w4²] (krpm²)
        """
        Mx, My, Mz = torque
        
        # 首先进行X Y轴分配（不考虑Z轴）
        Mz_temp = 0
        control_input = np.array([thrust, Mx, My, Mz_temp])
        motor_speed_sq = self.inv_mat @ control_input
        
        # 检查饱和情况
        max_value = np.max(motor_speed_sq)
        min_value = np.min(motor_speed_sq)
        ref_value = np.mean(motor_speed_sq)
        
        # 计算缩放因子
        max_scale = 1.0
        min_scale = 1.0
        
        if max_value > self.max_speed ** 2:
            max_scale = (self.max_speed ** 2 - ref_value) / (max_value - ref_value)
        
        if min_value < 0:
            min_scale = ref_value / (ref_value - min_value)
        
        scale = min(max_scale, min_scale)
        
        # 缩放X Y扭矩
        Mx_scaled = Mx * scale
        My_scaled = My * scale
        
        # 重新计算（仍然没有Z轴）
        control_input = np.array([thrust, Mx_scaled, My_scaled, Mz_temp])
        motor_speed_sq = self.inv_mat @ control_input
        
        if scale < 1.0:
            # 存在饱和，不分配Z轴扭矩
            motor_speed_sq = np.abs(motor_speed_sq)
            return np.sqrt(motor_speed_sq)
        else:
            # 有余量，分配Z轴扭矩
            control_input_with_z = np.array([thrust, Mx_scaled, My_scaled, Mz])
            motor_speed_sq_with_z = self.inv_mat @ control_input_with_z
            
            # 检查饱和
            max_value_z = np.max(motor_speed_sq_with_z)
            min_value_z = np.min(motor_speed_sq_with_z)
            
            # 找出最大和最小值对应的索引
            max_index = np.argmax(motor_speed_sq_with_z)
            min_index = np.argmin(motor_speed_sq_with_z)
            
            max_scale_z = 1.0
            min_scale_z = 1.0
            
            if max_value_z > self.max_speed ** 2:
                max_scale_z = (self.max_speed ** 2 - motor_speed_sq[max_index]) / (max_value_z - motor_speed_sq[max_index])
            
            if min_value_z < 0:
                min_scale_z = motor_speed_sq[min_index] / (motor_speed_sq[min_index] - min_value_z)
            
            scale_z = min(max_scale_z, min_scale_z)
            
            # 缩放Z轴扭矩
            Mz_scaled = Mz * scale_z
            
            # 最终计算
            control_input_final = np.array([thrust, Mx_scaled, My_scaled, Mz_scaled])
            motor_speed_sq_final = self.inv_mat @ control_input_final
            
            # 确保非负
            motor_speed_sq_final = np.abs(motor_speed_sq_final)
            
            return np.sqrt(motor_speed_sq_final)
    
    def krpm_to_thrust(self, krpm: np.ndarray) -> np.ndarray:
        """将电机转速转换为推力"""
        return self.Ct * krpm ** 2
    
    def thrust_to_krpm(self, thrust: np.ndarray) -> np.ndarray:
        """将推力转换为电机转速"""
        return np.sqrt(np.maximum(thrust / self.Ct, 0))
    
    def krpm_to_torque(self, krpm: np.ndarray) -> np.ndarray:
        """将电机转速转换为扭矩"""
        return self.Cd * krpm ** 2
    
    def normalize_input(self, krpm: np.ndarray) -> np.ndarray:
        """将电机转速归一化到[0, 1]范围"""
        return np.clip(krpm / self.max_speed, 0, 1)
    
    def __str__(self):
        return f"Mixer(L={self.L}m, Ct={self.Ct:.6f}, Cd={self.Cd:.6f}, layout={self.motor_layout})"