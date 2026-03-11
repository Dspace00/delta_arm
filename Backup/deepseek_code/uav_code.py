# # Note


# all rotation related is used with scipy
# MuJoCo quaternion is defined as [w, x, y, z]
# quaternion is defined as [w, x, y, z]
# Max_thrust = 6.5 N                        per_prop
# Max_torque = 0.15 Nm                      per_prop
# Max_krpm = 22
# Ct = Max_thrust / krpm^2 = 6.5 / 22 /22 = 0.013429  N / krpm^2
# Cm = Max_torque / krpm^2 = 0.15 / 22 / 22 = 3.099e-4 Nm / krpm^2
# 
# Top View
# prop1/prop3: anti_clockwise
# prop2/prop4: clockwise


import mujoco
import mujoco.viewer
import numpy as np

import quaternion
import math
# quaternion [w, x, y, z]
import time
import os
# import geometry
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple

def generate_circular_trajectory(
    center: List[float],  # 圆心坐标 [x, y, z]
    radius: float,        # 半径
    total_time: float,    # 总时间
    num_points: int = 50, # 轨迹点数（默认50个点）
    start_angle: float = 0,  # 起始角度（弧度，默认从右侧开始）
    clockwise: bool = False, # 是否顺时针
    height_variation: bool = False,  # Z坐标是否随角度变化
    height_amplitude: float = 0.5,   # Z坐标变化幅度
    hold_last: bool = True           # 是否在最后保持位置
) -> List[Tuple[float, List[float]]]:
    """
    生成圆周运动轨迹
    
    参数:
        center: 圆心坐标 [x, y, z]
        radius: 圆的半径
        total_time: 总运动时间
        num_points: 轨迹点数
        start_angle: 起始角度（弧度）
        clockwise: True为顺时针，False为逆时针
        height_variation: 是否让Z坐标随角度变化
        height_amplitude: Z坐标变化幅度
        hold_last: 是否在最后时刻保持位置（添加额外的停留点）
    
    返回:
        轨迹列表，每个元素为 (时间, [x, y, z])
    """
    
    trajectory = []
    direction = -1 if clockwise else 1  # 方向：顺时针为-1，逆时针为1
    
    # 计算时间步长
    time_step = total_time / (num_points - 1) if num_points > 1 else total_time
    
    # 生成轨迹点
    for i in range(num_points):
        # 计算当前时间
        t = i * time_step
        
        # 计算当前角度（绕圆心转动的角度）
        if num_points > 1:
            angle = start_angle + direction * 2 * math.pi * i / (num_points - 1)
        else:
            angle = start_angle
        
        # 计算当前位置
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        
        # 计算Z坐标
        if height_variation:
            # Z坐标随角度变化（正弦变化）
            z = center[2] + height_amplitude * math.sin(angle * 2)  # 可以调整频率
        else:
            z = center[2]
        
        trajectory.append((t, [x, y, z]))
    
    # 如果需要在最后保持位置
    if hold_last and trajectory:
        last_point = trajectory[-1]
        # 添加一个稍后的时间点，保持相同位置
        hold_time = last_point[0] + 1.0  # 保持1秒
        trajectory.append((hold_time, last_point[1]))
    
    return trajectory


def generate_spiral_trajectory(
    center: List[float],    # 起始中心点
    start_radius: float,    # 起始半径
    end_radius: float,      # 结束半径
    total_time: float,      # 总时间
    num_turns: int = 2,     # 螺旋圈数
    num_points: int = 100,  # 总点数
    clockwise: bool = False # 是否顺时针
) -> List[Tuple[float, List[float]]]:
    """
    生成螺旋轨迹（半径逐渐变化）
    """
    trajectory = []
    direction = -1 if clockwise else 1
    
    for i in range(num_points):
        t = i * total_time / (num_points - 1) if num_points > 1 else 0
        
        # 当前角度（多圈）
        angle = direction * 2 * math.pi * num_turns * i / (num_points - 1)
        
        # 当前半径（线性变化）
        current_radius = start_radius + (end_radius - start_radius) * i / (num_points - 1)
        
        # 计算位置
        x = center[0] + current_radius * math.cos(angle)
        y = center[1] + current_radius * math.sin(angle)
        z = center[2]
        
        trajectory.append((t, [x, y, z]))
    
    return trajectory

class Mixer:
    def __init__(self):

        self.Ct = 0.01343       # 电机推力系数 (N/krpm^2) 注意结果单位为力(N)
        self.Cd = 3.099e-4     # 电机反扭系数 (Nm/krpm^2) 注意结果单位为扭矩(Nm)
        self.L = 0.18           # 电机力臂长度 单位m
        self.max_thrust = 6.5   # 单个电机最大推力 单位N (电机最大转速22krpm)
        self.max_torque = 0.15  # 单个电机最大扭矩 单位Nm (电机最大转速22krpm)
        self.max_speed = 22     # 电机最大转速(krpm)

        # 动力分配正向矩阵
        self.mat = np.array([
            [self.Ct, self.Ct, self.Ct, self.Ct],                                   # F total
            [self.Ct*self.L, -self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L],     # Mx + - - +
            [-self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L, self.Ct*self.L],     # My - - + +
            [-self.Cd, self.Cd, -self.Cd, self.Cd]                                  # Mz - + - +
        ])
        # 动力分配逆向矩阵
        self.inv_mat = np.linalg.inv(self.mat)

    # 动力分配
    # thrust: 机体总推力 单位N
    # mx, my, mz: 三轴扭矩 单位Nm
    def calculate(self, thrust, mx, my, mz):
        Mx, My = mx, my  # Copy
        Mz = 0 # 首先进行X Y轴分配
        control_input = np.array([thrust, Mx, My, Mz])
        motor_speed_squ = self.inv_mat @ control_input
        # X Y Z三轴动力分配的顺序决定最终取舍的不同
        # 一般情况下 首先对X Y轴动力进行分配 余量用于分配Z轴
        max_value = np.max(motor_speed_squ)
        min_value = np.min(motor_speed_squ)
        ref_value = np.sum(motor_speed_squ) / 4.0  # 参考转速(不施加扭矩时的转速平方)
        # print(f"ref_value:{ref_value}")
        max_trim_scale = 1.0
        min_trim_scale = 1.0
        if max_value > self.max_speed **2: # 存在电机动力饱和 计算缩放因子进行缩放
            # print(f"Max Overflow")
            max_trim_scale = (self.max_speed ** 2 - ref_value)/(max_value - ref_value)
        if min_value < 0: # 存在电机动力负饱和 计算缩放因子进行缩放
            # print(f"Min Overflow")
            min_trim_scale = (ref_value)/(ref_value - min_value)
        scale = min(max_trim_scale, min_trim_scale)
        # print(f"Trim Scale:{scale}")
        # 对X Y扭矩施加缩放因子
        Mx = Mx * scale  
        My = My * scale
        # 重新计算电机转速平方
        control_input = np.array([thrust, Mx, My, Mz])
        motor_speed_squ = self.inv_mat @ control_input
        # print(f"motor_speed_squ:{motor_speed_squ}")
        # print(f"Original Torque: Mx:{Mx/scale:.6f} My:{My/scale:.6f} Trimed Torque: Mx:{Mx:.6f} My:{My:.6f}")
        if scale < 1.0: # 存在Trim 不进行Z轴扭矩分配 直接返回
            # 这里需要强行进行一下绝对值
            motor_speed_squ = np.abs(motor_speed_squ)
            return np.sqrt(motor_speed_squ)  # 返回电机转速
        else: # 仍然有余量 可以进行Z轴扭矩分配
            Mz = mz
            control_input_withz = np.array([thrust, Mx, My, Mz])  # 添加Z轴扭矩重新计算
            motor_speed_squ_withz = self.inv_mat @ control_input_withz
            # 判断是否饱和
            max_value = np.max(motor_speed_squ_withz)
            min_value = np.min(motor_speed_squ_withz)
            max_index = np.argmax(motor_speed_squ_withz)
            min_index = np.argmin(motor_speed_squ_withz)
            max_trim_scale_z = 1.0
            min_trim_scale_z = 1.0
            if max_value > self.max_speed **2: # 存在电机动力饱和 计算缩放因子进行缩放
                # print(f"Z Max Overflow")
                max_trim_scale_z = (self.max_speed ** 2 - motor_speed_squ[max_index])/(max_value - motor_speed_squ[max_index])
            if min_value < 0: # 存在电机动力负饱和 计算缩放因子进行缩放
                # print(f"Z Min Overflow")
                min_trim_scale_z = (motor_speed_squ[min_index])/(motor_speed_squ[min_index] - min_value)
            scale_z = min(max_trim_scale_z, min_trim_scale_z)
            # 对Z轴扭矩施加缩放因子
            Mz = Mz * scale_z
            # 重新计算电机转速平方
            control_input_withz = np.array([thrust, Mx, My, Mz])
            motor_speed_squ_withz = self.inv_mat @ control_input_withz
            # print(f"motor_speed_squ:{motor_speed_squ_withz}")
            # print(f"Original Torque: Mx:{Mx/scale:.6f} My:{My/scale:.6f} Mz:{Mz/scale_z:.6f} Trimed Torque: Mx:{Mx:.6f} My:{My:.6f} Mz:{Mz:.6f}")
            motor_speed_squ = np.abs(motor_speed_squ)
            return np.sqrt(motor_speed_squ_withz)  # 返回电机转速

    def calc_motor_force(self, krpm):
        return self.Ct * krpm**2

    # 根据推力计算电机转速
    def calc_motor_speed_by_force(self, force):

        if force > self.max_thrust:
            force = self.max_thrust
        elif force < 0:
            force = 0
        return np.sqrt(force / self.Ct)

    # 根据扭矩计算电机转速 注意返回数值为转速绝对值 根据实际情况决定转速是增加还是减少
    def calc_motor_speed_by_torque(self, torque):

        if torque > self.max_torque:  # 扭矩绝对值限制
            torque = self.max_torque
        return np.sqrt(torque / self.Cd)

    # 根据电机转速计算电机转速
    def calc_motor_speed(self, force):
        if force > 0:
            return Mixer.calc_motor_speed_by_force(self, force)

    # 根据电机转速计算电机扭矩
    def calc_motor_torque(self, krpm):
        return self.Cd * krpm**2

    # 根据电机转速计算电机归一化输入
    def calc_motor_input(self, krpm):
        if krpm > 22:
            krpm = 22
        elif krpm < 0:
            krpm = 0
        _force = self.calc_motor_force(krpm)
        _input = _force / self.max_thrust
        if _input > 1:
            _input = 1
        elif _input < 0:
            _input = 0
        return _input

class DiscretePID3D:
    """3D离散PID控制器类"""
    
    def __init__(self, kp, ki, kd, dt, output_limits=(-float('inf'), float('inf'))):
        """
        初始化3D PID控制器
        
        参数:
            kp: 比例增益 [3,] 或 标量
            ki: 积分增益 [3,] 或 标量
            kd: 微分增益 [3,] 或 标量
            dt: 采样时间
            output_limits: 输出限制 (min, max)
        """
        self.kp = np.array(kp) if hasattr(kp, '__len__') else np.ones(3) * kp
        self.ki = np.array(ki) if hasattr(ki, '__len__') else np.ones(3) * ki
        self.kd = np.array(kd) if hasattr(kd, '__len__') else np.ones(3) * kd
        self.dt = dt
        
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.output_limits = output_limits
        
    def update(self, setpoint, current_value):
        """更新PID控制器并返回控制输出"""
        # 计算误差
        error = np.array(setpoint) - np.array(current_value)
        
        # 比例项
        proportional = self.kp * error
        
        # 积分项 (带抗饱和)
        self.integral += error * self.dt
        integral_term = self.ki * self.integral
        
        # 微分项
        derivative = (error - self.previous_error) / self.dt
        derivative_term = self.kd * derivative
        
        # 计算输出
        output = proportional + integral_term + derivative_term
        
        # 应用输出限制
        if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # 更新上次误差
        self.previous_error = error
        # self.integral = np.zeros(3)

        return output
    
    def reset(self):
        """重置控制器状态"""
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
## ----------------------------------------------------------------------------------------------------------------- ##
class PosPID:
    """3D离散PID控制器类"""
    
    def __init__(self, kp, dt, output_limits=(-float('inf'), float('inf'))):
        """
        初始化3D PID控制器
        
        参数:
            kp: 比例增益 [3,] 或 标量
            ki: 积分增益 [3,] 或 标量
            kd: 微分增益 [3,] 或 标量
            dt: 采样时间
            output_limits: 输出限制 (min, max)
        """
        self.kp = np.array(kp) if hasattr(kp, '__len__') else np.ones(3) * kp
        self.dt = dt
        
        self.output_limits = output_limits
        
    def update(self, setpoint, current_value):
        """更新PID控制器并返回控制输出"""
        # 计算误差
        error = np.array(setpoint) - np.array(current_value)
                
        # 计算输出
        output = self.kp * error
        
        # 应用输出限制
        if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
            output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return output
## ----------------------------------------------------------------------------------------------------------------- ##
class VelPID:
    """3D离散PID控制器类"""
    
    def __init__(self, kp, ki, kd, dt, output_limits=(-float('inf'), float('inf'))):
        """
        初始化3D PID控制器
        
        参数:
            kp: 比例增益 [3,] 或 标量
            ki: 积分增益 [3,] 或 标量
            kd: 微分增益 [3,] 或 标量
            dt: 采样时间
            output_limits: 输出限制 (min, max)
        """
        self.kp = np.array(kp) if hasattr(kp, '__len__') else np.ones(3) * kp
        self.ki = np.array(ki) if hasattr(ki, '__len__') else np.ones(3) * ki
        self.kd = np.array(kd) if hasattr(kd, '__len__') else np.ones(3) * kd
        self.dt = dt
        
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.Vel_PID_last_value = np.zeros(3)
        self.output_limits = output_limits
        
    def update(self, setpoint, current_value):
        """更新PID控制器并返回控制输出"""
        # 计算误差
        error = np.array(setpoint) - np.array(current_value)
        
        # 比例项
        proportional = self.kp * error
        
        # 积分项 (带抗饱和)
        self.integral += error * self.dt
        integral_term = self.ki * self.integral
        
        # 微分项
        v_dot = (current_value - self.Vel_PID_last_value) / self.dt
        self.Vel_PID_last_value = current_value
        # derivative = (error - self.previous_error) / self.dt
        derivative_term = self.kd * (0 - v_dot)

        # 计算输出
        output = proportional + integral_term + derivative_term
        
        # 应用输出限制
        if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # 更新上次误差
        self.previous_error = error
        # self.integral = np.zeros(3)

        return output
    
    def reset(self):
        """重置控制器状态"""
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.Vel_PID_last_value = np.zeros(3)
## ----------------------------------------------------------------------------------------------------------------- ##
class AttitudePID3D:
    """3D离散PID控制器类"""
    
    def __init__(self, kp, dt, output_limits=(-float('inf'), float('inf'))):
        """
        初始化3D PID控制器
        
        参数:
            kp: 比例增益 [3,] 或 标量
            ki: 积分增益 [3,] 或 标量
            kd: 微分增益 [3,] 或 标量
            dt: 采样时间
            output_limits: 输出限制 (min, max)
        """
        self.kp = np.array(kp) if hasattr(kp, '__len__') else np.ones(3) * kp

        self.dt = dt
        
        self.output_limits = output_limits

    ## ------------------------------    
    # def update(self, setpoint, current_value):
    #     # calculate desired angular velocity from two quaternion
    #     """更新PID控制器并返回控制输出"""
    #     # 计算误差

    #     cur_conj = np.quaternion(current_value[0], -current_value[1], -current_value[2], -current_value[3])
    #     tar_quat = np.quaternion(setpoint[0], setpoint[1], setpoint[2], setpoint[3])

    #     quat_error = cur_conj * tar_quat

    #     error = np.array([quat_error.x, quat_error.y, quat_error.z])
        
    #     # 比例项
    #     proportional = self.kp * error
        
    #     # 积分项 (带抗饱和)
    #     self.integral += error * self.dt
    #     integral_term = self.ki * self.integral
        
    #     # 微分项
    #     derivative = (error - self.previous_error) / self.dt
    #     derivative_term = self.kd * derivative
        
    #     # 计算输出
    #     output = proportional + integral_term + derivative_term
        
    #     # 应用输出限制
    #     if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
    #         output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
    #     # 更新上次误差
    #     self.previous_error = error
    #     # self.integral = np.zeros(3)

    #     return output
    
## AI new
    def update(self, setpoint, current_value):
        """更新PID控制器并返回控制输出"""
        # 计算误差 - 将setpoint和current_value都转换为quaternion对象
        # 检查setpoint类型
        if isinstance(setpoint, np.quaternion):
            tar_quat = setpoint
        elif isinstance(setpoint, (list, tuple, np.ndarray)) and len(setpoint) == 4:
            tar_quat = np.quaternion(setpoint[0], setpoint[1], setpoint[2], setpoint[3])
        else:
            raise ValueError("setpoint必须是np.quaternion或长度为4的数组")
        
        # 检查current_value类型
        if isinstance(current_value, np.quaternion):
            cur_quat = current_value
            print("cur_quat is np.quat")
        elif isinstance(current_value, (list, tuple, np.ndarray)) and len(current_value) == 4:
            cur_quat = np.quaternion(current_value[0], current_value[1], 
                                    current_value[2], current_value[3])
        else:
            raise ValueError("current_value必须是np.quaternion或长度为4的数组")
        
        # print(f"tar_quat: {tar_quat}, cur_quat: {cur_quat}")
        # 计算误差四元数：q_error = q_current^-1 * q_target
        cur_conj = cur_quat.conjugate()  # 使用conjugate()方法更清晰
        quat_error = tar_quat * cur_conj
        
        # 提取虚部作为误差向量（这对应旋转轴*sin(theta/2)）
        error = np.array([quat_error.x, quat_error.y, quat_error.z])
        
        # 注意：如果四元数表示旋转，通常需要确保w>=0以表示最短路径
        if quat_error.w < 0:
            error = -error  # 取反以保持最短路径
        
        # 比例项
        proportional = self.kp * error
        
        # 计算输出
        output = proportional
        
        # 应用输出限制
        if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        return output             
## ----------------------------------------------------------------------------------------------------------------- ##
class AngVelPID:
    """3D离散PID控制器类"""
    
    def __init__(self, kp, ki, kd, dt, output_limits=(-float('inf'), float('inf'))):
        """
        初始化3D PID控制器
        
        参数:
            kp: 比例增益 [3,] 或 标量
            ki: 积分增益 [3,] 或 标量
            kd: 微分增益 [3,] 或 标量
            dt: 采样时间
            output_limits: 输出限制 (min, max)
        """
        self.kp = np.array(kp) if hasattr(kp, '__len__') else np.ones(3) * kp
        self.ki = np.array(ki) if hasattr(ki, '__len__') else np.ones(3) * ki
        self.kd = np.array(kd) if hasattr(kd, '__len__') else np.ones(3) * kd
        self.dt = dt
        
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.output_limits = output_limits
        self.AngVel_PID_last_value = np.zeros(3)

    # def update(self, setpoint, current_value):
    #     """更新PID控制器并返回控制输出"""
    #     # 计算误差
    #     error = np.array(setpoint) - np.array(current_value)
        
    #     # 比例项
    #     proportional = self.kp * error
        
    #     # 积分项 (带抗饱和)
    #     self.integral += error * self.dt
    #     integral_term = self.ki * self.integral
        
    #     # 微分项
    #     derivative = (error - self.previous_error) / self.dt
    #     derivative_term = self.kd * derivative
        
    #     # 计算输出
    #     output = proportional + integral_term + derivative_term
        
    #     # 应用输出限制
    #     if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
    #         output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
    #     # 更新上次误差
    #     self.previous_error = error
    #     # self.integral = np.zeros(3)

    #     return output
    
    def update(self, setpoint, current_value):
        """更新PID控制器并返回控制输出"""
        # 计算误差
        error = np.array(setpoint) - np.array(current_value)
        
        # 比例项
        proportional = self.kp * error
        
        # 积分项 (带抗饱和)
        self.integral += error * self.dt
        integral_term = self.ki * self.integral
        
        # 微分项
        v_dot = (current_value - self.AngVel_PID_last_value) / self.dt
        self.AngVel_PID_last_value = current_value
        # derivative = (error - self.previous_error) / self.dt
        derivative_term = self.kd * (0 - v_dot)

        # 计算输出
        output = proportional + integral_term + derivative_term
        
        # 应用输出限制
        if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # 更新上次误差
        self.previous_error = error
        # self.integral = np.zeros(3)

        return output

    def reset(self):
        """重置控制器状态"""
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.AngVel_PID_last_value = np.zeros(3)
    


class QuadrotorCascadePID:
    """四旋翼无人机串级PID控制器"""
    
    def __init__(self, model, data):
        # MuJoCo模型和数据
        self.model = model
        self.data = data
        
        # 仿真参数
        self.sim_dt = model.opt.timestep  # 仿真时间步长
        self.sim_time = 0.0
        
        # 控制器频率 (与pid.py保持一致)
        self.position_freq = 50     # Hz
        self.velocity_freq = 50     # Hz
        self.attitude_freq = 250    # Hz
        self.angle_rate_freq = 1000 # Hz
        
        # 计算各控制器的更新时间步长
        self.position_dt = 1.0 / self.position_freq
        self.velocity_dt = 1.0 / self.velocity_freq
        self.attitude_dt = 1.0 / self.attitude_freq
        self.angle_rate_dt = 1.0 / self.angle_rate_freq
        
        # 上次更新时间
        self.last_position_update = 0.0
        self.last_velocity_update = 0.0
        self.last_attitude_update = 0.0
        self.last_angle_rate_update = 0.0
        
        # 无人机物理参数 (需要根据实际模型调整)
        self.mass = 1.27  # kg
        self.gravity = 9.81
        self.arm_length = 0.18  # 机臂长度
        self.thrust_coeff = 1  # 推力系数
        self.torque_coeff = 0.15  # 扭矩系数

        self.total_thrust_cmd = 0.0
        
        # 初始化PID控制器 (参数需要根据具体系统调整)
        self.position_pid = PosPID([8.0, 8.0, 12.0], self.position_dt)
        self.velocity_pid = VelPID([4, 4, 6], [0.0, 0.0, 0.0], [0.05, 0.05, 0.05], self.velocity_dt)
        self.attitude_pid = AttitudePID3D([9, 9, 12], self.attitude_dt)
        self.angle_rate_pid = AngVelPID([3, 3, 3], [100.0, 100.0, 100.0], [0.00005, 0.00005, 0.00005], self.angle_rate_dt)
        
        # 目标状态
        self.target_position = np.array([0.0, 0.0, 1.0])  # 目标位置 [x, y, z]
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # 目标速度
        self.target_attitude = np.quaternion(1, 0, 0, 0) # [w, x, y, z]
        self.target_angle_rate = np.array([0.0, 0.0, 0.0])  # 目标角速度

        # 控制输出
        self.motor_thrusts = np.zeros(4)  # 四个电机的推力
        self.controller_ouput_torque = np.zeros(3)  # 三轴扭矩命令
        self.total_thrust = np.zeros(3)

        self.controller_mix = np.zeros(6)
        # 数据记录
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'targeet_velocity': [],
            'eular': [],
            'target_position': [],
            'target_eular': [],
            'angle_rate': [],
            'target_angle_rate': [],
            'motor_thrusts': [],
            'motor_mix': []
        }
    
    def get_drone_state(self):
        """获取无人机当前状态"""
        # 假设无人机主体名为"uav_body"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "UAV_body")
        
        # if body_id == -1:
        #     # 如果找不到指定名称，使用第一个主体
        #     body_id = 0
        
        # 位置和四元数
        position = self.data.body(body_id).xpos.copy()
        quaternion = self.data.body(body_id).xquat.copy() # [w,x,y,z] default mujoco quat settings
        body_att_mat = self.data.body(body_id).xmat.copy().reshape(3,3)
        # 转换为欧拉角
        rotation_matrix = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]], scalar_first = True)
        # euler_angles = rotation.as_euler('ZYX')  # [yaw, pitch, roll]
        # attitude = R.as_quat(rotation, scalar_first = True)  # [x,y,z,w] change order of quaternion
        
        # use quaternion describe attitude
        # attitude = np.quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        attitude = quaternion
        # 线速度和角速度
        velocity = self.data.body(body_id).cvel[3:6].copy()  # 线性速度
        angle_rate_world = self.data.body(body_id).cvel[0:3].copy()  # angular velocity in world frame
        # 将角速度转换到机体坐标系
        angle_rate = body_att_mat.T @ angle_rate_world

        return position, velocity, attitude, angle_rate
    
    def print_info(self):
        position, velocity, attitude, angle_rate = self.get_drone_state()
        np.set_printoptions(precision=4, suppress=True)
        sensor_data = self.data.sensordata.copy()

        sensor_quat = sensor_data[6:10]  # imu四元数 [w, x, y, z]
        sensor_angle_rate = sensor_data[0:3]  # imu角速度
        att_quat = np.array([self.target_attitude.w, self.target_attitude.x, self.target_attitude.y, self.target_attitude.z])
        error = attitude - att_quat

    
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Position: {position}")
        print(f"Velocity: {velocity}")
        print(f"Attitude (quat): {attitude}")
        print(f"Sensor Attitude (quat): {sensor_quat}")
        print(f"Angle Rate: {angle_rate}")
        print(f"Sensor Angle Rate: {sensor_angle_rate}")

        print(f"quat Error: {error}")
        print(f"Target Position: {self.target_position}")
        print(f"Target Velocity: {self.target_velocity}")
        print(f"Target Attitude (quat): {att_quat}")
        print(f"Target Angle Rate: {self.target_angle_rate}")
        print(f"Thrusts value: {self.total_thrust_cmd}")
        print(f"total Thrust Vector: {self.total_thrust}")
        print(f"Controller Output Torque: {self.controller_ouput_torque}")
        print(f"controller mix: {self.controller_mix}")

    def limit_acc_vec_degree(self, v):
        """
        调整向量使其与z轴夹角不小于30度
        参数:
            v: 三维向量 (x, y, z)，其中z > 0
        返回:
            调整后的三维向量
        """
        # 计算向量的模长
        magnitude = np.linalg.norm(v)
        
        # 计算向量与z轴的夹角（弧度）
        z_axis = np.array([0, 0, 1])
        dot_product = np.dot(v, z_axis)
        cos_theta = dot_product / (magnitude * np.linalg.norm(z_axis))
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        theta_deg = np.degrees(theta_rad)
        
        # 如果夹角小于30度，返回原向量
        if theta_deg < 30.0:
            return v, theta_deg, False
        
        # 需要旋转到刚好30度的夹角
        target_theta_rad = np.radians(30.0)
        rotation_angle_rad = theta_rad - target_theta_rad
        
        # 计算旋转轴（原向量与z轴的叉乘）
        rotation_axis = np.cross(v, z_axis)
        
        # 如果旋转轴长度为0（向量与z轴平行），使用x轴作为旋转轴
        if np.linalg.norm(rotation_axis) < 1e-10:
            rotation_axis = np.array([1, 0, 0])
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # 使用罗德里格斯旋转公式进行旋转
        cos_phi = np.cos(rotation_angle_rad)
        sin_phi = np.sin(rotation_angle_rad)
        
        # 旋转矩阵的组成
        ux, uy, uz = rotation_axis
        cross_matrix = np.array([
            [0, -uz, uy],
            [uz, 0, -ux],
            [-uy, ux, 0]
        ])
        
        outer_matrix = np.outer(rotation_axis, rotation_axis)
        identity_matrix = np.eye(3)
        
        # 罗德里格斯旋转矩阵
        rotation_matrix = cos_phi * identity_matrix + sin_phi * cross_matrix + (1 - cos_phi) * outer_matrix
        
        # 应用旋转
        rotated_vector = np.dot(rotation_matrix, v)
        
        # 保持模长不变（理论上应该不变，但数值计算可能有微小误差）
        rotated_vector = rotated_vector / np.linalg.norm(rotated_vector) * magnitude
        
        return rotated_vector, theta_deg, True

    def update_controllers(self):
        """根据各自频率更新各个控制器"""
        current_position, current_velocity, current_attitude, current_angle_rate = self.get_drone_state()
        ## ----------------------------------------------------------------------------------------------------------------- ##
        # 位置控制器 (50Hz)
        if self.sim_time >= self.last_position_update + self.position_dt:
            self.target_velocity = self.position_pid.update(self.target_position, current_position)
            self.last_position_update = self.sim_time
        ## ----------------------------------------------------------------------------------------------------------------- ##
        # 速度控制器 (50Hz)
        if self.sim_time >= self.last_velocity_update + self.velocity_dt:
            # 在速度控制中考虑重力补偿
            gravity_compensation = np.array([0, 0, -self.gravity])

            # Desired yaw angle
            yaw_angle = np.deg2rad(0) 
            yaw_axis = np.array([-np.sin(yaw_angle), np.cos(yaw_angle), 0]) # This is the body-y axis project to X-Y palne

            acc_pid_results = self.velocity_pid.update(self.target_velocity, current_velocity)
            
            acc_prop = acc_pid_results - gravity_compensation
            # print(f"acc_prop: {acc_prop}")


            # limit_acc_cmd, z_axis_angle, need_to_rotate = self.limit_acc_vec_degree(acc_prop)
            # total_thrust = self.mass * (limit_acc_cmd)
            
            # print(f"z_angle: {z_axis_angle}, need to rotate? :{need_to_rotate}")
            # 计算总推力和期望姿态
            # Total thrust in vector
            total_thrust = self.mass * (acc_prop)
            self.controller_mix[0:3] = total_thrust

            z_axis = total_thrust / np.linalg.norm(total_thrust)

            x_axis = np.cross(yaw_axis, z_axis)
            x_axis_norm = np.linalg.norm(x_axis)
            
            # if x_axis_norm < 1e-10:
            #     # 如果yaw_axis和z_axis平行，选择默认的x轴
            #     # 这里选择与z轴垂直的一个向量
            #     if abs(z_axis[2]) < 0.9:
            #         x_axis = np.array([z_axis[1], -z_axis[0], 0])
            #     else:
            #         x_axis = np.array([0, z_axis[2], -z_axis[1]])
            #     x_axis = x_axis / np.linalg.norm(x_axis)
            # else:
            #     x_axis = x_axis / x_axis_norm    
                     
            
            x_axis = x_axis / x_axis_norm
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            # 5. 构建旋转矩阵：列向量为[x_axis, y_axis, z_axis]
            attitude_mat = np.column_stack((x_axis, y_axis, z_axis))
            # print(f"mat: {attitude_mat}")
            self.target_attitude = quaternion.from_rotation_matrix(attitude_mat)
            # print(f"att: {att}")
            # set target quat for attitude calculation
            # self.target_attitude = R.as_quat(attitude_mat, scalar_first = True)
            # print(f"target_att: {self.target_attitude}")
            # 存储总推力用于后续计算
            self.total_thrust = total_thrust
            self.total_thrust_cmd = np.linalg.norm(total_thrust)
            
        ## ----------------------------------------------------------------------------------------------------------------- ##
        # 姿态控制器 (250Hz)
        if self.sim_time >= self.last_attitude_update + self.attitude_dt:
            self.target_angle_rate = self.attitude_pid.update(self.target_attitude, current_attitude)
            self.last_attitude_update = self.sim_time

        ## ----------------------------------------------------------------------------------------------------------------- ##
        # 角速度控制器 (1000Hz)
        if self.sim_time >= self.last_angle_rate_update + self.angle_rate_dt:
            torque_cmd = self.angle_rate_pid.update(self.target_angle_rate, current_angle_rate)
            self.controller_ouput_torque = torque_cmd
            # 将总推力和力矩转换为电机推力
            # self.motor_thrusts = self.thrust_mixing(self.total_thrust_cmd, torque_cmd)
            self.controller_mix[3:6] = torque_cmd
            # 应用电机推力
            self.apply_motor_thrusts()
            
            self.last_angle_rate_update = self.sim_time
    
    def thrust_mixing(self, total_thrust, torques):
        # 根据电机转速计算电机归一化输入
        # print(f"torques: {torques}")
        # max_thrust = 6.5

        # def calc_motor_force(krpm):
        #     global Ct
        #     return Ct * krpm**2

        # def calc_motor_input(krpm):
        #     if krpm > 22:
        #         krpm = 22
        #     elif krpm < 0:
        #         krpm = 0
        #     _force = calc_motor_force(krpm)
        #     _input = _force / max_thrust
        #     if _input > 1:
        #         _input = 1
        #     elif _input < 0:
        #         _input = 0
        #     return _input

        """将总推力和力矩分配到四个电机"""
        # 推力分配矩阵 (X型配置)
        # 电机顺序: 前左(0), 前右(1), 后右(2), 后左(3)
        L = self.arm_length
        c = self.thrust_coeff
        b = self.torque_coeff
        # mixer = Mixer()
        # motor_speed = mixer.calculate(total_thrust, torques[0], torques[1], torques[2]) # 动力分配
        # d.actuator('rotor1').ctrl[0] = calc_motor_input(motor_speed[0])
        # d.actuator('rotor2').ctrl[0] = calc_motor_input(motor_speed[1])
        # d.actuator('rotor3').ctrl[0] = calc_motor_input(motor_speed[2])
        # d.actuator('rotor4').ctrl[0] = calc_motor_input(motor_speed[3])
        # 构建分配矩阵
        mixing_matrix = np.array([
            [c, c, c, c],          # 总推力
            [c*L, -c*L, -c*L, c*L],         # 滚转力矩
            [-c*L, -c*L, c*L, c*L],         # 俯仰力矩
            [-b, b, -b, b]         # 偏航力矩
        ])
        
        # 计算所需的力量向量 [总推力, 滚转力矩, 俯仰力矩, 偏航力矩]
        # print(f"force: {total_thrust}, torque: {torques}")
        forces_torques = np.array([total_thrust, torques[0], torques[1], torques[2]])
        
        # 求解电机推力 (使用伪逆)
        motor_thrusts = np.linalg.inv(mixing_matrix) @ forces_torques
        
        # # 确保推力非负
        # motor_thrusts = np.maximum(motor_thrusts, 0)
        
        return motor_thrusts
    
    def apply_motor_thrusts(self):
        """将计算出的电机推力应用到MuJoCo执行器"""
        # 假设执行器名为"motor0"到"motor3"
        # for i in range(4):
        #     actuator_name = f"rotor{i}"
        #     actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            
        #     if actuator_id != -1:
        #         # self.data.ctrl[actuator_id] = self.motor_thrusts[i]
        #         self.data.actuator(f'rotor{i}').ctrl[0] = self.motor_thrusts[i]

        virtual_force_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "force")  
        att_mat = self.data.body("UAV_body").xmat.copy().reshape(3,3)
        force_body = att_mat.T @ self.controller_mix[0:3]

        self.data.actuator(f'forcex').ctrl[0] = force_body[0] 
        self.data.actuator(f'forcey').ctrl[0] = force_body[1]
        self.data.actuator(f'forcez').ctrl[0] = force_body[2]
        self.data.actuator(f'Mx').ctrl[0] = self.controller_mix[3]
        self.data.actuator(f'My').ctrl[0] = self.controller_mix[4]
        self.data.actuator(f'Mz').ctrl[0] = self.controller_mix[5]


    def set_target_position(self, position):
        """设置目标位置"""
        self.target_position = np.array(position)
        # print(f"tar_pos: {self.target_position}")
    
    def set_target_yaw(self, yaw):
        """设置目标偏航角"""
        self.target_attitude[2] = yaw
    
    def record_data(self):
        """记录当前状态数据"""
        position, velocity, attitude, angle_rate = self.get_drone_state()
        
        # Convert quaternion to euler angles for easier interpretation
        eular = R.from_quat(attitude, scalar_first = True).as_euler('ZYX', degrees=True)
        if abs(eular[0]) > 180 or abs(eular[1]) > 180 or abs(eular[2]) > 180:
            print(f"eular angle error! {eular}" )

        target_attitude = np.array([self.target_attitude.w, self.target_attitude.x, self.target_attitude.y, self.target_attitude.z])
        target_eular = R.from_quat(target_attitude, scalar_first = True).as_euler('ZYX', degrees=True)
        self.history['time'].append(self.sim_time)
        self.history['position'].append(position.copy())
        self.history['velocity'].append(velocity.copy())
        self.history['targeet_velocity'].append(self.target_velocity.copy())
        self.history['eular'].append(eular.copy())
        self.history['target_eular'].append(target_eular)
        self.history['target_position'].append(self.target_position.copy())
        self.history['angle_rate'].append(angle_rate.copy())
        self.history['target_angle_rate'].append(self.target_angle_rate.copy())
        self.history['motor_thrusts'].append(self.motor_thrusts.copy())
        self.history['motor_mix'].append(self.controller_mix.copy())
    
    def reset(self):
        """重置控制器状态"""
        self.position_pid.reset()
        self.velocity_pid.reset()
        self.attitude_pid.reset()
        self.angle_rate_pid.reset()
        
        self.sim_time = 0.0
        self.last_position_update = 0.0
        self.last_velocity_update = 0.0
        self.last_attitude_update = 0.0
        self.last_angle_rate_update = 0.0

class OptimizedDroneSimulation:
    """优化的无人机仿真类"""
    
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 时间参数
        self.sim_timestep = self.model.opt.timestep
        self.target_render_fps = 60
        self.render_interval = 1.0 / self.target_render_fps
        self.steps_per_render = max(1, int(self.render_interval / self.sim_timestep))
        
        # 初始化PID控制器
        self.controller = QuadrotorCascadePID(self.model, self.data)
        
        # 状态变量
        self.step_count = 0
        
        self.display_sim_info()
    
    def display_sim_info(self):
        """显示仿真信息"""
        print(f"仿真步长: {self.sim_timestep} s")
        print(f"目标渲染频率: {self.target_render_fps} Hz")
        print(f"渲染间隔: {self.render_interval:.4f} s")
        print(f"每 {self.steps_per_render} 步渲染一次")
        print(f"实际渲染频率: {1.0/(self.steps_per_render * self.sim_timestep):.1f} Hz")
        print(f"控制器频率 - 位置: {self.controller.position_freq}Hz, 姿态: {self.controller.attitude_freq}Hz, 角速度: {self.controller.angle_rate_freq}Hz")
    
    def run_simulation(self, duration=10.0, target_trajectory=None):
        """运行仿真"""
        total_steps = int(duration / self.sim_timestep)
        frame_count = 0

        scene_option = mujoco.MjvOption()      
        scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        # 默认轨迹: 悬停然后移动
        if target_trajectory is None:
            target_trajectory = [
                (0.0, [0, 0, 1]),      # 初始位置
                # (2.0, [1, 0, 1.5]),    # 2秒后移动到新位置
                # (5.0, [0, 1, 2]),      # 5秒后
                # (8.0, [-1, 0, 1.5]),   # 8秒后
                (10.0, [0, 0, 1])      # 返回原点
            ]
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("开始无人机仿真...")
            start_time = time.time()
            
            for step in range(total_steps):
                current_time = step * self.sim_timestep
                self.controller.sim_time = current_time
                
                # 更新目标轨迹
                for i in range(len(target_trajectory) - 1):
                    t_start, pos_start = target_trajectory[i]
                    t_end, pos_end = target_trajectory[i + 1]
                    
                    if t_start <= current_time < t_end:
                        alpha = (current_time - t_start) / (t_end - t_start)
                        target_pos = pos_start + alpha * (np.array(pos_end) - np.array(pos_start))
                        self.controller.set_target_position(target_pos)
                        break
                # 更新控制器
                # self.controller.set_target_position(np.array([0.0, 0.0, 0.5]))

                self.controller.update_controllers()
                
                # 执行仿真步
                mujoco.mj_step(self.model, self.data)
                
                
                # time.sleep(0.001)

                # 记录数据 (降低频率以减少数据量)
                if step % 10 == 0:
                    self.controller.record_data()
                    # self.controller.print_info()

                # if step % 1000 == 0:
                #     # self.controller.record_data()
                #     self.controller.print_info()
                # 控制渲染频率
                if step % self.steps_per_render == 0:
                    viewer.sync()
                    frame_count += 1
                
                # 打印进度
                if step % 5000 == 0:
                    elapsed = time.time() - start_time
                    progress = step / total_steps * 100
                    pos = self.controller.get_drone_state()[0]
                    # print(f"进度: {progress:.1f}%, 位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            
            # 最终统计
            elapsed_time = time.time() - start_time
            actual_fps = frame_count / elapsed_time
            
            print(f"\n=== 仿真完成 ===")
            print(f"总步数: {total_steps}")
            print(f"总帧数: {frame_count}")
            print(f"实际运行时间: {elapsed_time:.2f}s")
            print(f"实际渲染FPS: {actual_fps:.1f} Hz")
            print(f"实时因子: {duration / elapsed_time:.2f}x")
    
    def plot_results(self):
        """绘制仿真结果 (需要matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            history = self.controller.history
            time_array = np.array(history['time'])
            position = np.array(history['position'])
            target_position = np.array(history['target_position'])
            target_eular = np.array(history['target_eular'])
            eular = np.array(history['eular'])
            motor_thrusts = np.array(history['motor_thrusts'])
            motor_mix = np.array(history['motor_mix'])
            
            vel = np.array(history['velocity'])
            target_vel = np.array(history['targeet_velocity'])

            angle_rate = np.array(history['angle_rate'])
            target_angle_rate = np.array(history['target_angle_rate'])

            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            
            # 位置跟踪
            axes[0, 0].plot(time_array, position[:, 0], 'r-', label='X')
            axes[0, 0].plot(time_array, position[:, 1], 'g-', label='Y')
            axes[0, 0].plot(time_array, position[:, 2], 'b-', label='Z')
            axes[0, 0].plot(time_array, target_position[:, 0], 'r--', label='Target X')
            axes[0, 0].plot(time_array, target_position[:, 1], 'g--', label='Target Y')
            axes[0, 0].plot(time_array, target_position[:, 2], 'b--', label='Target Z')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].set_title('Position Tracking')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[1, 0].plot(time_array, vel[:, 0], 'r-', label='X')
            axes[1, 0].plot(time_array, vel[:, 1], 'g-', label='Y')
            axes[1, 0].plot(time_array, vel[:, 2], 'b-', label='Z')
            axes[1, 0].plot(time_array, target_vel[:, 0], 'r--', label='Target X')
            axes[1, 0].plot(time_array, target_vel[:, 1], 'g--', label='Target Y')
            axes[1, 0].plot(time_array, target_vel[:, 2], 'b--', label='Target Z')
            axes[1, 0].set_ylabel('Vel (m/s)')
            axes[1, 0].set_title('Vel Tracking')
            axes[1, 0].legend()
            axes[1, 0].grid(True)            

            # 姿态
            axes[0, 2].plot(time_array, eular[:, 0], 'r-', label='Yaw')
            axes[0, 2].plot(time_array, target_eular[:, 0], 'b--', label='Target Yaw')
            axes[0, 2].set_ylabel('yaw (deg)')
            axes[0, 2].set_title('Drone eular Angles')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
            axes[0, 2].set_ylim([-180, 180])


            axes[1, 2].plot(time_array, eular[:, 1], 'r-', label='Pitch')
            axes[1, 2].plot(time_array, target_eular[:, 1], 'b--', label='Target Pitch')         
            axes[1, 2].set_ylabel('pitch (deg)')
            axes[1, 2].set_title('Drone eular Angles')
            axes[1, 2].legend()
            axes[1, 2].grid(True)


            axes[2, 2].plot(time_array, eular[:, 2], 'r-', label='Roll')
            axes[2, 2].plot(time_array, target_eular[:, 2], 'b--', label='Target Roll')          
            axes[2, 2].set_ylabel('Roll (deg)')
            axes[2, 2].set_title('Drone eular Angles')
            axes[2, 2].legend()
            axes[2, 2].grid(True)

            # angle rates
            axes[0, 3].plot(time_array, angle_rate[:, 0], 'r-', label='Yaw')
            axes[0, 3].plot(time_array, target_angle_rate[:, 0], 'b--', label='Target Yaw')
            axes[0, 3].set_ylabel('Roll (deg)')
            axes[0, 3].set_title('Drone Angles Velocity')
            axes[0, 3].legend()
            axes[0, 3].grid(True)

            axes[1, 3].plot(time_array, angle_rate[:, 1], 'r-', label='Pitch')
            axes[1, 3].plot(time_array, target_angle_rate[:, 1], 'b--', label='Target Pitch')         
            axes[1, 3].set_ylabel('pitch (deg)')
            axes[1, 3].set_title('Drone Angles Velocity')
            axes[1, 3].legend()
            axes[1, 3].grid(True)


            axes[2, 3].plot(time_array, angle_rate[:, 2], 'r-', label='Roll')
            axes[2, 3].plot(time_array, target_angle_rate[:, 2], 'b--', label='Target Roll')          
            axes[2, 3].set_ylabel('yaw (deg)')
            axes[2, 3].set_title('Drone Angles Velocity')
            axes[2, 3].legend()
            axes[2, 3].grid(True)

            axes[0, 1].plot(time_array, motor_mix[:, 0], 'r-', label='Fx')
            axes[0, 1].plot(time_array, motor_mix[:, 1], 'g-', label='Fy')
            axes[0, 1].plot(time_array, motor_mix[:, 2], 'b-', label='Fz')
            axes[0, 1].set_ylabel('Force (N)')
            axes[0, 1].set_title('Total Thrust Vector')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            axes[1, 1].plot(time_array, motor_mix[:, 3], 'r-', label='Mx')
            axes[1, 1].plot(time_array, motor_mix[:, 4], 'g-', label='My')
            axes[1, 1].plot(time_array, motor_mix[:, 5], 'b-', label='Mz')
            axes[1, 1].set_ylabel('Torques (Nm)')
            axes[1, 1].set_title('Total Torques')
            axes[1, 1].legend()
            axes[1, 1].grid(True)            

            # 电机推力
            for i in range(4):
                axes[2, 1].plot(time_array, motor_thrusts[:, i], label=f'Motor {i}')
            axes[2, 1].set_ylabel('Thrust (N)')
            axes[2, 1].set_title('Motor Thrusts')
            axes[2, 1].legend()
            axes[2, 1].grid(True)
            
            # 3D轨迹
            ax_3d = fig.add_subplot(4, 4, 14, projection='3d')
            ax_3d.plot(position[:, 0], position[:, 1], position[:, 2], 'b-', label='Actual')
            ax_3d.plot(target_position[:, 0], target_position[:, 1], target_position[:, 2], 'r--', label='Target')
            ax_3d.quiver(0, 0, 0, 15, 0, 0, color='gray', alpha=0.5, linestyle='dashed', arrow_length_ratio=0.3)
            ax_3d.quiver(0, 0, 0, 0, 15, 0, color='gray', alpha=0.5, linestyle='dashed', arrow_length_ratio=0.3)
            ax_3d.quiver(0, 0, 0, 0, 0, 15, color='gray', alpha=0.5, linestyle='dashed', arrow_length_ratio=0.3)
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title('3D Trajectory')
            ax_3d.legend()
            
            # 位置误差
            position_error = np.linalg.norm(position - target_position, axis=1)
            axes[2, 0].plot(time_array, position_error, 'k-')
            axes[2, 0].set_ylabel('Position Error (m)')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].set_title('Position Error')
            axes[2, 0].grid(True)

            # 速度误差
            vel_error = np.linalg.norm(vel - target_vel, axis=1)
            axes[3, 0].plot(time_array, vel_error, 'm-')
            axes[3, 0].set_ylabel('Vel Error (m/s)')
            axes[3, 0].set_xlabel('Time (s)')
            axes[3, 0].set_title('Vel Error')
            axes[3, 0].grid(True)            
            # 空子图，可用于其他数据
            axes[3, 1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib未安装，无法绘制图形")

# 使用示例
if __name__ == "__main__":
    # 运行仿真
    simulator = OptimizedDroneSimulation("Delta.xml")  # 
    
    # 定义目标轨迹: (时间, [x, y, z])
    # trajectory = [
    #     (0.0, [0, 0, 1.5]),
    #     (1.5, [0, 0, 1.5]),
    #     (3.0, [1, 0, 1.5]),
    #     (6.0, [1, 1, 2.0]),
    #     (9.0, [1, -1, 1.5]),
    #     (12.0, [-1, -1, 1.0]),
    #     (15.0, [-1, 1, 1.0]),

    #     (20.0, [-5, 0, 1.5])
    # ]
    
    trajectory = [
        (0.0, [0, 0, 0.5]),
        (2.0, [0, 0, 1.0]),
        (4.0, [1, 0, 1.0]),
        (6.0, [1, 1, 1.0]),
        (8.0, [0, 1, 1.0]),
        (10.0, [0, 0, 1.0]),
        (50.0, [0, 0, 1.0])
    ]

    trajectory = [
        (0.0, [0, 0, 1.0]),
        (20, [0, 0, 1.0])
    ]
        
    center = [0, 0, 1.0]  # 圆心坐标
    radius = 1.0          # 半径
    total_time = 10.0     # 总时间10秒
    
    trajectory1 = generate_circular_trajectory(
        center=center,
        radius=radius,
        total_time=total_time,
        num_points=130,    # 生成13个点（包括起点和终点）
        start_angle=0,    # 从右侧开始
        clockwise=False,   # 逆时针
        height_variation=False  # Z坐标不变
    )

    trajectory2 = generate_circular_trajectory(
        center=[0, 0, 0.8],
        radius=2.5,
        total_time=10.0,
        num_points=200,
        height_variation=True,
        height_amplitude=0.3
    )

    trajectory3 = generate_spiral_trajectory(
        center=[0, 0, 0.5],
        start_radius=0.5,
        end_radius=5.0,
        total_time=10.0,
        num_turns=3,
        num_points=150
    )

    simulator.run_simulation(duration=10, target_trajectory=trajectory1)
    simulator.plot_results()