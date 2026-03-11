import time
import mujoco
import mujoco.viewer
import numpy as np
import socket
import json
import threading
from typing import List, Tuple, Optional


class DeltaRobot:
    """Delta机械臂模型封装类"""
    
    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._setup_model_indices()

        self.s_P = 0.02495 * np.sqrt(3)  # 平台等边三角形边长
        self.L = 0.1        # 上臂长度
        self.l = 0.2        # 下臂平行四边形长度
        
        # 其他几何参数（表1）
        self.w_B = 0.074577      # 从{0}到基座近边的平面距离
        self.u_B = 2 * self.w_B  # 从{0}到基座顶点的平面距离   
        self.u_P = 0.02495        # 从{P}到平台顶点的平面距离
        self.w_P = 0.5 * self.u_P # 从{P}到平台近边的平面距离

        # 计算常数a, b, c（PDF第7页）
        self.a = self.w_B - self.u_P
        self.b = 0.5 * self.s_P - (np.sqrt(3)/2) * self.w_B
        self.c = self.w_P - 0.5 * self.w_B
        
        # 基座点坐标（PDF第6页）
        self.B1 = np.array([0, -self.w_B, 0])
        self.B2 = np.array([(np.sqrt(3)/2) * self.w_B, 0.5 * self.w_B, 0])
        self.B3 = np.array([-(np.sqrt(3)/2) * self.w_B, 0.5 * self.w_B, 0])
        
        # 平台点坐标（PDF第6页）
        self.P1 = np.array([0, -self.u_P, 0])
        self.P2 = np.array([0.5 * self.s_P, self.w_P, 0])
        self.P3 = np.array([-0.5 * self.s_P, self.w_P, 0])
        
        # 工作空间限制
        self.workspace_limits = {
            'x': [-0.15, 0.15],
            'y': [-0.15, 0.15], 
            'z': [-0.25, -0.05]
        }
            
    def inverse_kinematics(self, target_pos):
        """
        逆运动学计算 - 基于PDF第8-9页的方法
        输入: 目标位置 [x, y, z] (米)
        输出: 三个关节角度 [theta1, theta2, theta3] (弧度)
        """
        x, y, z = target_pos
        
        # 计算三个臂的系数（PDF第9页方程）
        E1 = 2 * self.L * (y + self.a)
        F1 = 2 * self.L * z
        G1 = (x**2 + y**2 + z**2 + self.a**2 + self.L**2 + 
              2*y*self.a - self.l**2)
        
        E2 = -self.L * (np.sqrt(3)*(x + self.b) + y + self.c)
        F2 = 2 * self.L * z
        G2 = (x**2 + y**2 + z**2 + self.b**2 + self.c**2 + self.L**2 + 
              2*x*self.b + 2*y*self.c - self.l**2)
        
        E3 = self.L * (np.sqrt(3)*(x - self.b) - y - self.c)
        F3 = 2 * self.L * z
        G3 = (x**2 + y**2 + z**2 + self.b**2 + self.c**2 + self.L**2 - 
              2*x*self.b + 2*y*self.c - self.l**2)
        
        # 使用三角代换求解（PDF第9页）
        def solve_arm(E, F, G):
            """求解单个臂的关节角度"""
            # 使用半角正切代换
            discriminant = F**2 + E**2 - G**2
            
            if discriminant < 0:
                raise ValueError("目标位置不可达")
            
            t1 = (-F + np.sqrt(discriminant)) / (G - E)
            t2 = (-F - np.sqrt(discriminant)) / (G - E)
            
            theta1 = 2 * np.arctan(t1)
            theta2 = 2 * np.arctan(t2)
            
            # 选择肘关节向外的解（knee outside）
            return theta2  # 根据PDF描述选择第二个解
        
        try:
            theta1 = solve_arm(E1, F1, G1)
            theta2 = solve_arm(E2, F2, G2)
            theta3 = solve_arm(E3, F3, G3)
            
            return [theta1, theta2, theta3]
            
        except ValueError as e:
            print(f"逆运动学错误: {e}")
            return None
    
    def forward_kinematics(self, joint_angles):
        """
        正运动学计算 - 基于PDF附录的三球相交算法
        输入: 三个关节角度 [theta1, theta2, theta3] (弧度)
        输出: 末端位置 [x, y, z] (米)
        """
        theta1, theta2, theta3 = joint_angles
        
        # 计算上臂端点（PDF第7页方程2）
        L1 = np.array([0, -self.L * np.cos(theta1), -self.L * np.sin(theta1)])
        L2 = np.array([
            (np.sqrt(3)/2) * self.L * np.cos(theta2),
            0.5 * self.L * np.cos(theta2),
            -self.L * np.sin(theta2)
        ])
        L3 = np.array([
            -(np.sqrt(3)/2) * self.L * np.cos(theta3),
            0.5 * self.L * np.cos(theta3),
            -self.L * np.sin(theta3)
        ])
        
        A1 = self.B1 + L1
        A2 = self.B2 + L2
        A3 = self.B3 + L3
        
        # 计算虚拟球心（PDF第7页）
        A1v = A1 - self.P1
        A2v = A2 - self.P2
        A3v = A3 - self.P3
        
        # 三球相交算法（PDF附录第37-38页）
        x1, y1, z1 = A1v
        x2, y2, z2 = A2v
        x3, y3, z3 = A3v
        
        # 计算系数（PDF第37页）
        a11 = 2 * (x3 - x1)
        a12 = 2 * (y3 - y1)
        a13 = 2 * (z3 - z1)
        b1 = -(x1**2 + y1**2 + z1**2) + (x3**2 + y3**2 + z3**2)
        
        a21 = 2 * (x3 - x2)
        a22 = 2 * (y3 - y2)
        a23 = 2 * (z3 - z2)
        b2 = -(x2**2 + y2**2 + z2**2) + (x3**2 + y3**2 + z3**2)
        
        # 求解z = f(x,y)（PDF第37-38页）
        if abs(a13) < 1e-10 or abs(a23) < 1e-10:
            raise ValueError("奇异位置，无法求解正运动学")
        
        a1 = a11/a13 - a21/a23
        a2 = a12/a13 - a22/a23
        a3 = b2/a23 - b1/a13
        
        a4 = -a2/a1 if abs(a1) > 1e-10 else 0
        a5 = -a3/a1 if abs(a1) > 1e-10 else 0
        
        a6 = (-a21*a4 - a22)/a23
        a7 = (b2 - a21*a5)/a23
        
        # 构建二次方程（PDF第38页）
        A_coeff = a4**2 + 1 + a6**2
        B_coeff = (2*a4*(a5 - x1) - 2*y1 + 2*a6*(a7 - z1))
        C_coeff = (a5*(a5 - 2*x1) + a7*(a7 - 2*z1) + 
                  x1**2 + y1**2 + z1**2 - self.l**2)
        
        # 求解y（PDF第38页方程A.10）
        discriminant = B_coeff**2 - 4*A_coeff*C_coeff
        
        if discriminant < 0:
            raise ValueError("三球无实交点")
        
        y_plus = (-B_coeff + np.sqrt(discriminant)) / (2*A_coeff)
        y_minus = (-B_coeff - np.sqrt(discriminant)) / (2*A_coeff)
        
        # 计算对应的x和z
        x_plus = a4*y_plus + a5
        z_plus = a6*y_plus + a7
        
        x_minus = a4*y_minus + a5
        z_minus = a6*y_minus + a7
        
        # 选择基座下方的解（PDF第8页）
        if z_plus > z_minus:  # z坐标更负的在下方的位置
            return np.array([x_plus, y_plus, z_plus])
        else:
            return np.array([x_minus, y_minus, z_minus])
    
    def check_workspace(self, position):
        """检查位置是否在工作空间内"""
        x, y, z = position
        limits = self.workspace_limits
        
        if (limits['x'][0] <= x <= limits['x'][1] and
            limits['y'][0] <= y <= limits['y'][1] and
            limits['z'][0] <= z <= limits['z'][1]):
            return True
        return False            
        
    def _setup_model_indices(self):
        """设置模型组件索引"""
        # 执行器索引
        self.motor_actuator_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor1'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor2'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor3')
        ]
        
        # 关节索引
        self.motor_joint_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'motor_arm_joint1'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'motor_arm_joint2'), 
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'motor_arm_joint3')
        ]
        
        # 关节位置地址
        self.motor_qpos_adr = [self.model.joint(i).qposadr[0] for i in self.motor_joint_indices]
        
         # 关节速度地址
        self.motor_qvel_adr = [self.model.joint(i).dofadr[0] for i in self.motor_joint_indices]

        # 末端平台body ID
        self.end_platform_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'end_platform')

    def get_joint_angles(self) -> np.ndarray:
        """获取当前关节角度"""
        return np.array([self.data.qpos[adr] for adr in self.motor_qpos_adr])

    def get_joint_velocities(self) -> np.ndarray:
        """获取关节角速度"""
        return np.array([self.data.qvel[adr] for adr in self.motor_qpos_adr])

    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置"""
        return self.data.xpos[self.end_platform_id].copy()
    
    def get_end_effector_velocity(self) -> np.ndarray:
        """获取末端执行器速度"""
        return self.data.cvel[self.end_platform_id][3:6].copy()
    
    def set_motor_velocities(self, velocities: np.ndarray):
        """
        设置电机角速度
        
        参数:
            velocities: 电机角速度 [ω1, ω2, ω3] (rad/s)
        """
        for i, velocity in enumerate(velocities):
            if i < len(self.motor_actuator_indices):
                self.data.ctrl[self.motor_actuator_indices[i]] = velocity

    def step(self):
        """执行一个仿真步"""
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        """重置仿真"""
        mujoco.mj_resetData(self.model, self.data)

# class LinearTrajectoryGenerator:
#     """
#     直线轨迹生成器
    
#     功能：输入起点和终点坐标，生成时间相关的直线轨迹和速度曲线
#     """
    
#     def __init__(self, max_velocity: float = 1.0, max_acceleration: float = 2.0):
#         """
#         初始化轨迹生成器
        
#         参数:
#             max_velocity: 最大速度 (m/s)
#             max_acceleration: 最大加速度 (m/s²)
#         """
#         self.max_velocity = max_velocity
#         self.max_acceleration = max_acceleration
        
#         # 轨迹参数
#         self.start_pos = None
#         self.target_pos = None
#         self.total_distance = 0.0
#         self.direction = None
#         self.trajectory_time = 0.0
        
#         # 时间参数
#         self.acceleration_time = 0.0
#         self.constant_velocity_time = 0.0
#         self.deceleration_time = 0.0
        
#     def set_trajectory(self, start_pos: np.ndarray, target_pos: np.ndarray, 
#                       move_time: Optional[float] = None) -> float:
#         """
#         设置轨迹参数
        
#         参数:
#             start_pos: 起点坐标 (3D)
#             target_pos: 终点坐标 (3D)
#             move_time: 期望移动时间 (秒)，如果为None则自动计算
            
#         返回:
#             实际移动时间
#         """
#         self.start_pos = np.array(start_pos, dtype=float)
#         self.target_pos = np.array(target_pos, dtype=float)
        
#         # 计算位移向量和距离
#         displacement = self.target_pos - self.start_pos
#         self.total_distance = np.linalg.norm(displacement)
#         self.direction = displacement / self.total_distance if self.total_distance > 0 else np.zeros(3)
        
#         if move_time is None:
#             # 自动计算最优移动时间（梯形速度曲线）
#             self.trajectory_time = self._calculate_optimal_time()
#         else:
#             self.trajectory_time = move_time
#             self._calculate_trapezoidal_profile()
            
#         return self.trajectory_time
    
#     def _calculate_optimal_time(self) -> float:
#         """计算梯形速度曲线的最优时间"""
#         # 计算达到最大速度所需的时间和距离
#         t_acc = self.max_velocity / self.max_acceleration
#         s_acc = 0.5 * self.max_acceleration * t_acc ** 2
        
#         if 2 * s_acc > self.total_distance:
#             # 三角形速度曲线（没有匀速段）
#             t_total = 2 * np.sqrt(self.total_distance / self.max_acceleration)
#             self.acceleration_time = t_total / 2
#             self.constant_velocity_time = 0.0
#             self.deceleration_time = t_total / 2
#         else:
#             # 梯形速度曲线
#             s_constant = self.total_distance - 2 * s_acc
#             t_constant = s_constant / self.max_velocity
#             t_total = 2 * t_acc + t_constant
#             self.acceleration_time = t_acc
#             self.constant_velocity_time = t_constant
#             self.deceleration_time = t_acc
            
#         return t_total
    
#     def _calculate_trapezoidal_profile(self):
#         """根据给定时间计算梯形速度曲线参数"""
#         # 简化的梯形速度曲线计算
#         # 在实际应用中可能需要更复杂的优化
#         self.acceleration_time = self.trajectory_time / 3
#         self.constant_velocity_time = self.trajectory_time / 3
#         self.deceleration_time = self.trajectory_time / 3
    
#     def get_desired_position(self, current_time: float) -> np.ndarray:
#         """
#         获取期望位置
        
#         参数:
#             current_time: 当前时间（从轨迹开始计时）
            
#         返回:
#             期望位置坐标
#         """
#         if self.start_pos is None or self.target_pos is None:
#             raise ValueError("请先调用 set_trajectory() 设置轨迹")
            
#         # 归一化时间 [0, 1]
#         t = np.clip(current_time / self.trajectory_time, 0.0, 1.0)
        
#         # 使用S曲线进行平滑插值（五次多项式）
#         s = self._s_curve_interpolation(t)
        
#         # 计算位置
#         position = self.start_pos + s * (self.target_pos - self.start_pos)
        
#         return position
    
#     def get_desired_velocity(self, current_time: float) -> np.ndarray:
#         """
#         获取期望速度
        
#         参数:
#             current_time: 当前时间（从轨迹开始计时）
            
#         返回:
#             期望速度向量
#         """
#         if self.start_pos is None or self.target_pos is None:
#             raise ValueError("请先调用 set_trajectory() 设置轨迹")
            
#         t = np.clip(current_time / self.trajectory_time, 0.0, 1.0)
        
#         # 计算S曲线的导数（速度）
#         ds_dt = self._s_curve_velocity(t)
        
#         # 速度大小
#         velocity_magnitude = ds_dt * (self.total_distance / self.trajectory_time)
        
#         # 速度向量
#         velocity = velocity_magnitude * self.direction
        
#         return velocity
    
#     def _s_curve_interpolation(self, t: float) -> float:
#         """
#         S曲线插值函数（五次多项式）
        
#         提供平滑的加速度和减速度
#         """
#         if t < 0:
#             return 0.0
#         elif t > 1:
#             return 1.0
#         else:
#             # 五次多项式：s(t) = 6t^5 - 15t^4 + 10t^3
#             return 6 * t**5 - 15 * t**4 + 10 * t**3
    
#     def _s_curve_velocity(self, t: float) -> float:
#         """
#         S曲线速度函数（五次多项式的导数）
#         """
#         if t < 0 or t > 1:
#             return 0.0
#         else:
#             # ds/dt = 30t^4 - 60t^3 + 30t^2
#             return 30 * t**4 - 60 * t**3 + 30 * t**2
    
#     def generate_trajectory_samples(self, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         生成轨迹采样点
        
#         参数:
#             num_samples: 采样点数量
            
#         返回:
#             (时间数组, 位置数组, 速度数组)
#         """
#         times = np.linspace(0, self.trajectory_time, num_samples)
#         positions = []
#         velocities = []
        
#         for t in times:
#             positions.append(self.get_desired_position(t))
#             velocities.append(self.get_desired_velocity(t))
            
#         return times, np.array(positions), np.array(velocities)
    
#     def get_trajectory_info(self) -> dict:
#         """获取轨迹信息"""
#         return {
#             'start_position': self.start_pos.tolist() if self.start_pos is not None else None,
#             'target_position': self.target_pos.tolist() if self.target_pos is not None else None,
#             'total_distance': self.total_distance,
#             'trajectory_time': self.trajectory_time,
#             'average_velocity': self.total_distance / self.trajectory_time if self.trajectory_time > 0 else 0,
#             'acceleration_time': self.acceleration_time,
#             'constant_velocity_time': self.constant_velocity_time,
#             'deceleration_time': self.deceleration_time
#         }

class UniformLinearTrajectory:
    """
    匀速直线轨迹生成器
    
    根据起始位置、目标位置和移动时间，生成匀速直线运动的轨迹
    """
    
    def __init__(self):
        # 轨迹参数
        self.start_pos = None
        self.target_pos = None
        self.move_time = 0.0
        self.velocity = None
        
        # 状态参数
        self.is_active = False
        self.start_time = 0.0
        
        # 计算参数
        self.total_distance = 0.0
        self.direction = None
        
    def set_trajectory(self, start_pos: np.ndarray, target_pos: np.ndarray, 
                      move_time: float) -> float:
        """
        设置轨迹参数
        
        参数:
            start_pos: 起始位置 (3D)
            target_pos: 目标位置 (3D)
            move_time: 移动时间 (秒)
            
        返回:
            移动时间
        """
        self.start_pos = np.array(start_pos, dtype=float)
        self.target_pos = np.array(target_pos, dtype=float)
        self.move_time = move_time
        
        # 计算位移向量和距离
        displacement = self.target_pos - self.start_pos
        self.total_distance = np.linalg.norm(displacement)
        
        if self.total_distance > 0:
            self.direction = displacement / self.total_distance
        else:
            self.direction = np.zeros(3)
            
        # 计算匀速运动的速度
        if move_time > 0:
            self.velocity = self.direction * (self.total_distance / move_time)
        else:
            self.velocity = np.zeros(3)
            
        self.is_active = True
        
        print(f"轨迹设置:")
        print(f"  起点: {self.start_pos}")
        print(f"  终点: {self.target_pos}")
        print(f"  距离: {self.total_distance:.4f}m")
        print(f"  时间: {move_time:.2f}s")
        print(f"  速度: {np.linalg.norm(self.velocity):.4f}m/s")
        
        return move_time
    
    def start(self, start_time: float):
        """
        开始轨迹跟踪
        
        参数:
            start_time: 轨迹开始时间
        """
        self.start_time = start_time
        self.is_active = True
        print(f"轨迹开始于时间: {start_time:.4f}s")
    
    def get_desired_state(self, current_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取期望的状态（位置和速度）
        
        参数:
            current_time: 当前时间
            
        返回:
            (期望位置, 期望速度)
        """
        if not self.is_active or self.start_pos is None:
            return self.start_pos if self.start_pos is not None else np.zeros(3), np.zeros(3)
        
        # 计算经过的时间
        elapsed_time = current_time - self.start_time
        
        if elapsed_time < 0:
            # 还未开始，保持在起始位置
            return self.start_pos, np.zeros(3)
        elif elapsed_time >= self.move_time:
            # 已经完成，保持在目标位置
            self.is_active = False
            return self.target_pos, np.zeros(3)
        else:
            # 运动中，匀速直线运动
            position = self.start_pos + self.velocity * elapsed_time
            return position, self.velocity
    
    def is_finished(self, current_time: float) -> bool:
        """检查轨迹是否完成"""
        if not self.is_active:
            return True
        elapsed_time = current_time - self.start_time
        return elapsed_time >= self.move_time
    
    def get_trajectory_info(self) -> dict:
        """获取轨迹信息"""
        return {
            'start_position': self.start_pos.tolist() if self.start_pos is not None else None,
            'target_position': self.target_pos.tolist() if self.target_pos is not None else None,
            'move_time': self.move_time,
            'velocity': self.velocity.tolist() if self.velocity is not None else None,
            'velocity_magnitude': np.linalg.norm(self.velocity) if self.velocity is not None else 0,
            'is_active': self.is_active,
            'start_time': self.start_time
        }

class CartesianTrajectoryController:
    """
    笛卡尔空间轨迹跟踪控制器
    
    使用PD控制计算速度误差，然后通过逆运动学转换为电机角速度
    """
    
    def __init__(self, robot, kp: float = 100.0, kd: float = 10.0):
        """
        初始化控制器
        
        参数:
            robot: DeltaRobot实例
            kp: 位置误差比例增益
            kd: 速度误差微分增益
        """
        self.robot = robot
        self.kp = kp
        self.kd = kd
        
        # 轨迹参数
        self.trajectory_generator = None
        self.trajectory_start_time = 0.0
        self.is_tracking = False
        
        # 控制参数
        self.control_freq = 100.0  # Hz
        self.control_dt = 1.0 / self.control_freq
        
    def set_trajectory(self, trajectory_generator, start_time: float):
        """
        设置要跟踪的轨迹
        
        参数:
            trajectory_generator: 轨迹生成器实例
            start_time: 轨迹开始时间
        """
        self.trajectory_generator = trajectory_generator
        self.trajectory_start_time = start_time
        self.is_tracking = True
        
    def update(self, current_time: float) -> Optional[np.ndarray]:
        """
        更新控制器，计算电机角速度
        
        参数:
            current_time: 当前仿真时间
            
        返回:
            电机角速度命令 (rad/s) 或 None (保持上一命令)
        """
        if not self.is_tracking or self.trajectory_generator is None:
            return None
            
        # 计算轨迹时间
        trajectory_time = current_time - self.trajectory_start_time
        
        # 检查轨迹是否完成
        if trajectory_time > self.trajectory_generator.move_time:
            self.is_tracking = False
            return np.zeros(3)  # 轨迹完成后停止
            
        # 获取期望的笛卡尔空间状态
        # desired_position = self.trajectory_generator.get_desired_position(trajectory_time)
        # desired_velocity = self.trajectory_generator.get_desired_velocity(trajectory_time)
        
        desired_position, desired_velocity = self.trajectory_generator.get_desired_state(trajectory_time)

        # 获取当前笛卡尔空间状态
        current_position = self.robot.get_end_effector_position()
        current_velocity = self.robot.get_end_effector_velocity()
        
        # 计算PD控制的速度误差项
        position_error = desired_position - current_position
        velocity_error = desired_velocity - current_velocity
        control_velocity_correction = self.kp * position_error + self.kd * velocity_error
        
        # 总控制速度 = 期望速度 + 速度修正
        control_velocity = desired_velocity + control_velocity_correction
        
        # 通过逆运动学将末端速度转换为电机角速度
        # 这里需要你的Delta机器人速度逆运动学实现
        motor_velocities = self.velocity_inverse_kinematics(control_velocity, current_position)
        
        return motor_velocities
    
    def velocity_inverse_kinematics(self, end_effector_velocity: np.ndarray, 
                                  current_position: np.ndarray) -> np.ndarray:
        """
        速度逆运动学 - 将末端平台速度转换为电机角速度
        
        参数:
            end_effector_velocity: 末端平台速度向量 [vx, vy, vz]
            current_position: 当前末端平台位置 [x, y, z]
            
        返回:
            电机角速度 [ω1, ω2, ω3] (rad/s)
            
        注意: 这是一个占位实现，你需要根据你的Delta机器人实际结构替换此方法
        """

        L_upper = self.robot.L  # 上臂长度
        L_lower = self.robot.l  # 下臂长度
        R = self.robot.w_B    # 基座近边距离
        r = self.robot.u_P    # 平台近边距离

        def phi_matrix(i):
            """计算旋转矩阵 phi_i (公式7)"""
            angle = (i) * np.pi / 3  # (i-1)π/3，这里i从0开始
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            phi_i = np.array([
                [cos_angle, sin_angle, 0],
                [-sin_angle, cos_angle, 0],
                [0, 0, 1]
            ])
            return phi_i
        
        def alpha_vector(q_i):
            """计算alpha向量 (公式9)"""
            return np.array([np.cos(q_i), 0, np.sin(q_i)])
        
        def d_alpha_dq(q_i):
            """计算alpha向量对q_i的导数"""
            return np.array([-np.sin(q_i), 0, np.cos(q_i)])
        
        def e_vector(q_i, i):
            """计算e_i向量 (公式9)"""
            alpha_i = alpha_vector(q_i)
            base_vector = np.array([R - r, 0, 0])
            e_i = base_vector + L_upper * alpha_i
            return e_i
        
        def beta_vector(current_position, q_i, i):
            """计算beta向量 (公式8)"""
            phi_i = phi_matrix(i)
            p_Ei_e = phi_i @ current_position  # 公式6
            e_i = e_vector(q_i, i)
            beta_i = (p_Ei_e - e_i) / L_lower
            return beta_i
                

        motor_angle = np.array(self.robot.inverse_kinematics(current_position))
        
        # 初始化M和V矩阵
        M_rows = []
        V_diag = []
        
        for i in range(3):
            # 计算beta_i
            beta_i = beta_vector(current_position, motor_angle[i], i)
            
            # 计算phi_i
            phi_i = phi_matrix(i)
            
            # 计算M矩阵的行 (公式12)
            M_row = beta_i.T @ phi_i
            M_rows.append(M_row)
            
            # 计算d_alpha/dq
            d_alpha_dq_val = d_alpha_dq(motor_angle[i])
            
            # 计算V矩阵的对角元素 (公式12)
            V_diag_element = beta_i.T @ d_alpha_dq_val
            V_diag.append(V_diag_element)
        
        # 构建M和V矩阵
        M = np.array(M_rows)
        V = np.diag(V_diag)
        
        # 计算电机角速度 (从公式11推导)
        # M * p_E_dot_e = L_upper * V * q_dot
        # => q_dot = (1/L_upper) * V^(-1) * M * p_E_dot_e
        
        try:
            V_inv = np.linalg.inv(V)
            q_dot = (1 / L_upper) * V_inv @ M @ end_effector_velocity
        except np.linalg.LinAlgError:
            # 如果V矩阵奇异，使用伪逆
            V_inv = np.linalg.pinv(V)
            q_dot = (1 / L_upper) * V_inv @ M @ end_effector_velocity
        
        return q_dot
    
    def stop(self):
        """停止轨迹跟踪"""
        self.is_tracking = False
        self.trajectory_generator = None

class DeltaVelocityController:
    """
    Delta机械臂速度控制器
    
    将电机角速度命令发送到MuJoCo
    """
    
    def __init__(self, robot, control_freq: float = 100.0):
        """
        初始化速度控制器
        
        参数:
            robot: DeltaRobot实例
            control_freq: 控制频率 (Hz)
        """
        self.robot = robot
        self.control_freq = control_freq
        self.sim_dt = robot.model.opt.timestep
        
        # 控制更新参数
        self.control_steps_per_update = int(1.0 / (control_freq * self.sim_dt))
        self.steps_since_last_control = 0
        
        # 当前控制命令
        self.current_motor_velocities = np.zeros(3)
        
        # 笛卡尔空间轨迹控制器
        self.cartesian_controller = CartesianTrajectoryController(robot)
        
    def set_cartesian_trajectory(self, trajectory_generator, start_time: float):
        """
        设置笛卡尔空间轨迹
        
        参数:
            trajectory_generator: 轨迹生成器实例
            start_time: 轨迹开始时间
        """
        self.cartesian_controller.set_trajectory(trajectory_generator, start_time)
    
    def set_motor_velocities(self, motor_velocities: np.ndarray):
        """
        直接设置电机角速度
        
        参数:
            motor_velocities: 电机角速度 [ω1, ω2, ω3] (rad/s)
        """
        self.current_motor_velocities = motor_velocities
        self.cartesian_controller.stop()  # 停止轨迹跟踪
    
    def update(self, current_time: float) -> Optional[np.ndarray]:
        """
        更新控制器
        
        参数:
            current_time: 当前仿真时间
            
        返回:
            电机角速度命令或None
        """
        self.steps_since_last_control += 1
        
        # 检查是否需要更新控制命令
        if self.steps_since_last_control >= self.control_steps_per_update:
            self.steps_since_last_control = 0
            
            # 更新笛卡尔空间控制器
            cartesian_control = self.cartesian_controller.update(current_time)
            if cartesian_control is not None:
                self.current_motor_velocities = cartesian_control
            
            return self.current_motor_velocities
        else:
            # 保持上一个控制命令
            return None
    
    def get_status(self) -> dict:
        """获取控制器状态"""
        return {
            'control_freq': self.control_freq,
            'sim_dt': self.sim_dt,
            'current_motor_velocities': self.current_motor_velocities.tolist(),
            'is_tracking_trajectory': self.cartesian_controller.is_tracking
        }

class StatusServer:
    """状态数据服务器"""
    
    def __init__(self, robot: DeltaRobot, host: str = 'localhost', port: int = 12348):
        self.robot = robot
        self.host = host
        self.port = port
        self.running = False
        self.status_socket = None
        
        # 状态数据模板
        self.status_data = {
            'joint_angles': [0, 0, 0],
            'joint_angles_deg': [0, 0, 0],
            'ee_position': [0, 0, 0],
            'ee_velocity': [0, 0, 0],
            'target_angles': [0, 0, 0],
            'target_position': [0, 0, 0],
            'is_moving': False
        }
    
    def start(self):
        """启动状态服务器"""
        self.running = True
        self.status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.status_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.status_socket.bind((self.host, self.port))
        self.status_socket.listen(5)
        
        print(f"状态服务器已启动在 {self.host}:{self.port}")
        
        server_thread = threading.Thread(target=self._handle_connections)
        server_thread.daemon = True
        server_thread.start()
    
    def stop(self):
        """停止状态服务器"""
        self.running = False
        if self.status_socket:
            self.status_socket.close()
    
    def update_status_data(self, target_positions: np.ndarray = None):

        """更新状态数据"""
        joint_angles = self.robot.get_joint_angles()
        ee_position = self.robot.get_end_effector_position()
        ee_velocity = self.robot.get_end_effector_velocity()

        # 运动状态判断（基于关节速度）
        is_moving = bool(np.any(np.abs(self.robot.get_joint_velocities()) > 1e-3))

        self.status_data.update({
            'joint_angles': joint_angles.tolist(),
            'joint_angles_deg': np.degrees(joint_angles).tolist(),
            'ee_position': ee_position.tolist(),
            'ee_velocity': ee_velocity.tolist(),
            'target_angles': (target_positions if target_positions is not None else joint_angles).tolist(),
            'target_position': target_positions.tolist(),  # 可根据需要修改为实际目标位置
            'is_moving': is_moving
        })
        # print("End Effector Position:", ee_position)
        # print("End Effector Velocity:", ee_velocity)
        # print("Joint Angles (rad):", joint_angles)
        # print("status_data updated:", self.status_data)
    
    def _handle_connections(self):
        """处理客户端连接"""
        while self.running:
            try:
                conn, addr = self.status_socket.accept()
                print(f"状态客户端连接: {addr}")
                
                # 为每个客户端创建独立线程
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(conn, addr)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"连接错误: {e}")
    
    def _handle_client(self, conn, addr):
        """处理单个客户端"""
        try:
            while self.running:
                # 发送状态数据
                data_str = json.dumps(self.status_data) + "\n"
                conn.sendall(data_str.encode())
                time.sleep(0.1)  # 10Hz 更新频率
                
        except Exception as e:
            print(f"客户端 {addr} 断开: {e}")
        finally:
            conn.close()

class DeltaRobotSimulation:
    """Delta机械臂仿真主类"""
    
    def __init__(self, model_path: str, control_freq: float = 100):
        self.robot = DeltaRobot(model_path)
        self.velocity_controller = DeltaVelocityController(self.robot, control_freq)
        self.control_freq = control_freq
        self.sim_dt = self.robot.model.opt.timestep
        
        # 创建状态服务器
        self.status_server = StatusServer(self.robot)
        
        # 仿真状态
        self.current_time = 0.0
        self.running = False
        
    def move_to_position(self, target_position: np.ndarray, move_time: float = 2.0):
        """
        移动到目标位置
        
        参数:
            target_position: 目标位置 [x, y, z]
            move_time: 移动时间
        """
        # 获取当前位置
        current_position = self.robot.get_end_effector_position()
        
        # 创建轨迹生成器
        trajectory_gen = UniformLinearTrajectory()
        trajectory_gen.set_trajectory(current_position, target_position, move_time)
        
        # 设置轨迹
        self.velocity_controller.set_cartesian_trajectory(trajectory_gen, self.current_time)
        
        print(f"开始移动到位置: {target_position}")
        print(f"轨迹时间: {move_time:.2f}s")
        
    def run(self, simulation_time: float = None):
        """运行仿真"""
        print("启动Delta机械臂仿真...")
        print(f"仿真步长: {self.robot.model.opt.timestep:.6f}s")
        print(f"控制频率: {self.velocity_controller.control_freq}Hz")
        
        # 启动状态服务器
        self.status_server.start()
        
        self.running = True
        self.current_time = 0.0
        frame_count = 0


        try:
            with mujoco.viewer.launch_passive(
                self.robot.model, self.robot.data
            ) as viewer:
                print("MuJoCo查看器已启动")
                # 主仿真循环
                while viewer.is_running and self.running:

                    step_start = time.time()
                    self.status_server.update_status_data(np.array([0.05, 0.0, -0.20]))                    
                    # 更新控制器
                    control_velocities = self.velocity_controller.update(self.current_time)
                    
                    # 应用控制命令
                    if control_velocities is not None:
                        self.robot.set_motor_velocities(control_velocities)
                    

                    
                    # 执行仿真步
                    self.robot.step()
                    self.current_time += self.robot.model.opt.timestep
                    
                    # 同步查看器
                    viewer.sync()
                    
                    # 实时仿真控制
                    elapsed = time.time() - step_start
                    sleep_time = self.robot.model.opt.timestep - elapsed

                    print(f"elapsed: {elapsed:.6f}s, sleep_time: {sleep_time:.6f}s")
                    # if sleep_time > 0:
                    #     time.sleep(sleep_time)
                    
                    # # 检查仿真时间限制
                    # if simulation_time and self.current_time >= simulation_time:
                    #     break
                        
        except Exception as e:
            print(f"仿真错误: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """停止仿真"""
        self.running = False
        self.status_server.stop()
        print("仿真已结束")


def main():
    """主函数"""
    # 创建仿真实例
    sim = DeltaRobotSimulation("/Users/yunxiaozang/Documents/GitHub/Delta_project/Delta/urdf/Delta.xml", control_freq=100)
    
    # 设置初始目标位置
    target_pos = np.array([0.05, 0.0, -0.20])
    # target_pos  = None
    sim.move_to_position(target_pos, move_time=1.0)
    
    # 运行仿真
    sim.run()


if __name__ == "__main__":
    main()