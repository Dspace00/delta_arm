"""
串级主控制器
整合位置、速度、姿态、角速度控制器
"""
import numpy as np
import quaternion
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import time

from ..utils import get_global_config
from .position_controller import PositionController
from .velocity_controller import VelocityController
from .attitude_controller import AttitudeController
from .angular_rate_controller import AngularRateController
from ..models.mixer import Mixer
from ..models.drone import DroneState


@dataclass
class ControlOutput:
    """控制输出数据类"""
    thrust_vector: np.ndarray  # 推力向量 [Fx, Fy, Fz]
    torque: np.ndarray         # 扭矩 [Mx, My, Mz]
    motor_speeds: np.ndarray   # 电机转速 [w1, w2, w3, w4]
    attitude_target: np.quaternion  # 目标姿态
    angular_rate_target: np.ndarray  # 目标角速度
    
    def __post_init__(self):
        # 确保数据正确性
        self.thrust_vector = np.asarray(self.thrust_vector)
        self.torque = np.asarray(self.torque)
        self.motor_speeds = np.asarray(self.motor_speeds)


# class CascadedController:
#     """串级主控制器"""
    
#     def __init__(self, config_manager=None, name: str = "Cascaded Controller"):
#         """
#         初始化串级控制器
        
#         Args:
#             config_manager: 配置管理器，如果为None则使用全局配置
#             name: 控制器名称
#         """
#         self.name = name
        
#         # 配置
#         if config_manager is None:
#             config_manager = get_global_config()
#         self.config = config_manager
        
#         # 无人机参数
#         self.mass = self.config.drone.mass
#         self.inertia = np.array([
#             self.config.drone.inertia['Ixx'],
#             self.config.drone.inertia['Iyy'],
#             self.config.drone.inertia['Izz']
#         ])
        
#         # 创建子控制器
#         self._create_subcontrollers()
        
#         # 创建混控器
#         self.mixer = Mixer(self.config.drone.motors)
        
#         # 目标状态
#         self.target_position = np.array([0.0, 0.0, 1.0])
#         self.target_velocity = np.zeros(3)
#         self.target_attitude = quaternion.from_float_array([1, 0, 0, 0])
#         self.target_angular_rate = np.zeros(3)
#         self.target_yaw = 0.0
        
#         # 控制输出
#         self.control_output = ControlOutput(
#             thrust_vector=np.zeros(3),
#             torque=np.zeros(3),
#             motor_speeds=np.zeros(4),
#             attitude_target=quaternion.from_float_array([1, 0, 0, 0]),
#             angular_rate_target=np.zeros(3)
#         )
        
#         # 控制模式
#         self.control_mode = self.config.controller.control_mode  # 'position', 'velocity', 'attitude', 'rate'
        
#         # 时间管理
#         self.sim_time = 0.0
#         self.control_frequencies = {
#             'position': self.config.controller.position['frequency'],
#             'velocity': self.config.controller.velocity['frequency'],
#             'attitude': self.config.controller.attitude['frequency'],
#             'angle_rate': self.config.controller.angle_rate['frequency']
#         }
        
#         # 上次更新时间
#         self.last_update_times = {
#             'position': 0.0,
#             'velocity': 0.0,
#             'attitude': 0.0,
#             'angle_rate': 0.0
#         }
        
#         # 状态历史记录
#         self.history = {
#             'time': [],
#             'position': [],
#             'velocity': [],
#             'attitude': [],
#             'angular_rate': [],
#             'target_position': [],
#             'target_velocity': [],
#             'target_attitude': [],
#             'target_angular_rate': [],
#             'thrust_vector': [],
#             'torque': [],
#             'motor_speeds': [],
#             'control_mode': []
#         }
        
#         print(f"Initialized {self.name}")
#         print(f"Control mode: {self.control_mode}")
#         print(f"Control frequencies: {self.control_frequencies}")
    
#     def _create_subcontrollers(self):
#         """创建子控制器"""
#         from .pid_factory import PIDControllerFactory
        
#         self.controllers = PIDControllerFactory.create_from_controller_config(
#             self.config.controller,
#             self.mass,
#             self.inertia
#         )
        
#         # 设置控制器名称
#         for name, controller in self.controllers.items():
#             controller.name = f"{self.name}_{name}"
    
#     def set_target(self, position=None, velocity=None, attitude=None, 
#                   angular_rate=None, yaw=None):
#         """
#         设置控制目标
        
#         Args:
#             position: 目标位置 [x, y, z]
#             velocity: 目标速度 [vx, vy, vz]
#             attitude: 目标姿态（四元数）
#             angular_rate: 目标角速度 [wx, wy, wz]
#             yaw: 目标偏航角（弧度）
#         """
#         if position is not None:
#             self.target_position = np.array(position)
        
#         if velocity is not None:
#             self.target_velocity = np.array(velocity)
        
#         if attitude is not None:
#             if isinstance(attitude, np.quaternion):
#                 self.target_attitude = attitude
#             else:
#                 self.target_attitude = quaternion.from_float_array(attitude)
        
#         if angular_rate is not None:
#             self.target_angular_rate = np.array(angular_rate)
        
#         if yaw is not None:
#             self.target_yaw = yaw
    
#     def update(self, drone_state: DroneState) -> ControlOutput:
#         """
#         更新控制器（主控制循环）
        
#         Args:
#             drone_state: 无人机当前状态
            
#         Returns:
#             控制输出
#         """
#         self.sim_time = drone_state.timestamp
        
#         # 根据控制模式执行相应的控制
#         if self.control_mode == 'position':
#             self._update_position_mode(drone_state)
#         elif self.control_mode == 'velocity':
#             self._update_velocity_mode(drone_state)
#         elif self.control_mode == 'attitude':
#             self._update_attitude_mode(drone_state)
#         elif self.control_mode == 'rate':
#             self._update_rate_mode(drone_state)
#         else:
#             raise ValueError(f"Unknown control mode: {self.control_mode}")
        
#         # 记录状态
#         self._record_state(drone_state)
        
#         return self.control_output
    
#     def _update_position_mode(self, drone_state: DroneState):
#         """位置控制模式"""
#         current_time = self.sim_time
        
#         # 位置控制器更新（频率较低）
#         position_dt = 1.0 / self.control_frequencies['position']
#         if current_time >= self.last_update_times['position'] + position_dt:
#             if 'position' in self.controllers:
#                 self.target_velocity = self.controllers['position'].update(
#                     self.target_position, drone_state.position
#                 )
#             self.last_update_times['position'] = current_time
        
#         # 调用速度控制模式
#         self._update_velocity_mode(drone_state)
    
#     def _update_velocity_mode(self, drone_state: DroneState):
#         """速度控制模式"""
#         current_time = self.sim_time
        
#         # 速度控制器更新
#         velocity_dt = 1.0 / self.control_frequencies['velocity']
#         if current_time >= self.last_update_times['velocity'] + velocity_dt:
#             if 'velocity' in self.controllers:
#                 # 计算加速度命令
#                 acceleration = self.controllers['velocity'].update(
#                     self.target_velocity, drone_state.velocity, velocity_dt
#                 )
                
#                 # 计算推力向量
#                 thrust_vector = self.controllers['velocity'].calculate_thrust_vector(acceleration)
                
#                 # 限制最大倾斜角度
#                 thrust_vector = self._limit_tilt_angle(thrust_vector)
                
#                 # 计算期望姿态
#                 if 'attitude' in self.controllers:
#                     self.target_attitude = self.controllers['attitude'].calculate_desired_attitude(
#                         thrust_vector, self.target_yaw
#                     )
                
#                 self.control_output.thrust_vector = thrust_vector
            
#             self.last_update_times['velocity'] = current_time
        
#         # 调用姿态控制模式
#         self._update_attitude_mode(drone_state)
    
#     def _update_attitude_mode(self, drone_state: DroneState):
#         """姿态控制模式"""
#         current_time = self.sim_time
        
#         # 姿态控制器更新
#         attitude_dt = 1.0 / self.control_frequencies['attitude']
#         if current_time >= self.last_update_times['attitude'] + attitude_dt:
#             if 'attitude' in self.controllers:
#                 self.target_angular_rate = self.controllers['attitude'].update(
#                     self.target_attitude, drone_state.attitude
#                 )
#                 self.control_output.attitude_target = self.target_attitude
            
#             self.last_update_times['attitude'] = current_time
        
#         # 调用角速度控制模式
#         self._update_rate_mode(drone_state)
    
#     def _update_rate_mode(self, drone_state: DroneState):
#         """角速度控制模式"""
#         current_time = self.sim_time
        
#         # 角速度控制器更新（最高频率）
#         rate_dt = 1.0 / self.control_frequencies['angle_rate']
#         if current_time >= self.last_update_times['angle_rate'] + rate_dt:
#             if 'angle_rate' in self.controllers:
#                 # 计算扭矩命令
#                 torque = self.controllers['angle_rate'].update(
#                     self.target_angular_rate, drone_state.angular_velocity, rate_dt
#                 )
#                 self.control_output.torque = torque
#                 self.control_output.angular_rate_target = self.target_angular_rate
            
#             self.last_update_times['angle_rate'] = current_time
        
#         # 动力分配（每次角速度控制都执行）
#         thrust_magnitude = np.linalg.norm(self.control_output.thrust_vector)
#         if thrust_magnitude > 0:
#             motor_speeds = self.mixer.allocate(thrust_magnitude, self.control_output.torque)
#             self.control_output.motor_speeds = motor_speeds
    
#     def _limit_tilt_angle(self, thrust_vector: np.ndarray, max_angle: float = 30.0) -> np.ndarray:
#         """
#         限制推力向量与垂直方向的夹角
        
#         Args:
#             thrust_vector: 推力向量
#             max_angle: 最大倾斜角度（度）
            
#         Returns:
#             限制后的推力向量
#         """
#         # 转换为垂直方向单位向量
#         vertical = np.array([0, 0, 1])
        
#         # 计算当前夹角
#         thrust_norm = np.linalg.norm(thrust_vector)
#         if thrust_norm < 1e-6:
#             return thrust_vector
        
#         thrust_dir = thrust_vector / thrust_norm
#         cos_angle = np.dot(thrust_dir, vertical)
#         angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
#         angle_deg = np.degrees(angle_rad)
        
#         # 如果夹角超过限制
#         if angle_deg > max_angle:
#             max_angle_rad = np.radians(max_angle)
            
#             # 计算旋转轴
#             rotation_axis = np.cross(vertical, thrust_dir)
#             axis_norm = np.linalg.norm(rotation_axis)
            
#             if axis_norm < 1e-6:
#                 # 平行的情况，选择任意垂直轴
#                 rotation_axis = np.array([1, 0, 0])
#             else:
#                 rotation_axis = rotation_axis / axis_norm
            
#             # 计算旋转角度
#             rotation_angle = angle_rad - max_angle_rad
            
#             # 使用罗德里格斯旋转公式
#             cos_r = np.cos(rotation_angle)
#             sin_r = np.sin(rotation_angle)
            
#             ux, uy, uz = rotation_axis
#             cross_matrix = np.array([
#                 [0, -uz, uy],
#                 [uz, 0, -ux],
#                 [-uy, ux, 0]
#             ])
            
#             outer_matrix = np.outer(rotation_axis, rotation_axis)
#             identity = np.eye(3)
            
#             rotation_matrix = cos_r * identity + sin_r * cross_matrix + (1 - cos_r) * outer_matrix
            
#             # 旋转推力方向
#             limited_dir = rotation_matrix @ thrust_dir
            
#             # 恢复原始大小
#             limited_thrust = limited_dir * thrust_norm
            
#             return limited_thrust
        
#         return thrust_vector
    
#     def _record_state(self, drone_state: DroneState):
#         """记录状态历史"""
#         self.history['time'].append(self.sim_time)
#         self.history['position'].append(drone_state.position.copy())
#         self.history['velocity'].append(drone_state.velocity.copy())
#         self.history['attitude'].append(drone_state.attitude.copy())
#         self.history['angular_rate'].append(drone_state.angular_velocity.copy())
#         self.history['target_position'].append(self.target_position.copy())
#         self.history['target_velocity'].append(self.target_velocity.copy())
#         self.history['target_attitude'].append(self.target_attitude.copy())
#         self.history['target_angular_rate'].append(self.target_angular_rate.copy())
#         self.history['thrust_vector'].append(self.control_output.thrust_vector.copy())
#         self.history['torque'].append(self.control_output.torque.copy())
#         self.history['motor_speeds'].append(self.control_output.motor_speeds.copy())
#         self.history['control_mode'].append(self.control_mode)
    
#     def reset(self):
#         """重置所有控制器状态"""
#         # 重置子控制器
#         for controller in self.controllers.values():
#             controller.reset()
        
#         # 重置状态
#         self.sim_time = 0.0
#         self.last_update_times = {
#             'position': 0.0,
#             'velocity': 0.0,
#             'attitude': 0.0,
#             'angle_rate': 0.0
#         }
        
#         # 清空历史
#         for key in self.history.keys():
#             self.history[key] = []
        
#         print(f"{self.name} reset")
    
#     def set_control_mode(self, mode: str):
#         """设置控制模式"""
#         valid_modes = ['position', 'velocity', 'attitude', 'rate']
#         if mode not in valid_modes:
#             raise ValueError(f"Invalid control mode: {mode}. Valid modes: {valid_modes}")
        
#         self.control_mode = mode
#         print(f"Control mode changed to: {mode}")
    
#     def get_status(self) -> Dict[str, Any]:
#         """获取控制器状态"""
#         return {
#             'control_mode': self.control_mode,
#             'target_position': self.target_position.tolist(),
#             'target_attitude': quaternion.as_float_array(self.target_attitude).tolist(),
#             'current_time': self.sim_time,
#             'subcontrollers': {name: str(ctrl) for name, ctrl in self.controllers.items()}
#         }
    
#     def save_history(self, filename: str = None):
#         """保存历史数据到文件"""
#         if filename is None:
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             filename = f"controller_history_{timestamp}.npz"
        
#         # 转换四元数为数组
#         attitude_history = np.array([quaternion.as_float_array(q) for q in self.history['attitude']])
#         target_att_history = np.array([quaternion.as_float_array(q) for q in self.history['target_attitude']])
        
#         np.savez_compressed(
#             filename,
#             time=np.array(self.history['time']),
#             position=np.array(self.history['position']),
#             velocity=np.array(self.history['velocity']),
#             attitude=attitude_history,
#             angular_rate=np.array(self.history['angular_rate']),
#             target_position=np.array(self.history['target_position']),
#             target_velocity=np.array(self.history['target_velocity']),
#             target_attitude=target_att_history,
#             target_angular_rate=np.array(self.history['target_angular_rate']),
#             thrust_vector=np.array(self.history['thrust_vector']),
#             torque=np.array(self.history['torque']),
#             motor_speeds=np.array(self.history['motor_speeds']),
#             control_mode=self.history['control_mode']
#         )
        
#         print(f"History saved to {filename}")
    
#     def __str__(self):
#         return f"CascadedController(name={self.name}, mode={self.control_mode}, " \
#                f"subcontrollers={list(self.controllers.keys())})"

    
class CascadedController:
    def __init__(self, config_manager=None, name: str = "Cascaded Controller"):
        # ... 现有初始化代码 ...
        
        # 添加：仿真步长（将在外部设置）
        self.sim_dt = 0.0001  # 默认值，会在集成时设置
        
        # 添加：步数计数器（基于步数而不是时间）
        self.step_count = 0
        
        # 计算各控制器的更新周期（单位：步数）
        # 位置：50Hz = 每0.02s → 0.02 / 0.0001 = 200步
        # 速度：50Hz = 每0.02s → 200步  
        # 姿态：250Hz = 每0.004s → 0.004 / 0.0001 = 40步
        # 角速度：1000Hz = 每0.001s → 0.001 / 0.0001 = 10步
        
        self.position_update_interval = int(0.02 / self.sim_dt)  # 200步
        self.velocity_update_interval = int(0.02 / self.sim_dt)  # 200步
        self.attitude_update_interval = int(0.004 / self.sim_dt)  # 40步
        self.angle_rate_update_interval = int(0.001 / self.sim_dt)  # 10步
        
        # 串级数据缓存（关键！）
        self._cached_target_velocity = np.zeros(3)
        self._cached_desired_attitude = quaternion.from_float_array([1, 0, 0, 0])
        self._cached_target_angular_rate = np.zeros(3)
        self._cached_thrust_vector = np.zeros(3)
        self._cached_torque = np.zeros(3)
        
    def set_simulation_timestep(self, dt: float):
        """设置仿真步长，重新计算更新间隔"""
        self.sim_dt = dt
        self.position_update_interval = int(0.02 / dt)  # 50Hz
        self.velocity_update_interval = int(0.02 / dt)  # 50Hz
        self.attitude_update_interval = int(0.004 / dt)  # 250Hz
        self.angle_rate_update_interval = int(0.001 / dt)  # 1000Hz
        
        print(f"控制器更新间隔设置：")
        print(f"  位置/速度：每 {self.position_update_interval} 步 (50Hz)")
        print(f"  姿态：每 {self.attitude_update_interval} 步 (250Hz)")
        print(f"  角速度：每 {self.angle_rate_update_interval} 步 (1000Hz)")
    
    def update(self, drone_state: DroneState) -> ControlOutput:
        """
        串级控制器更新 - 基于步数计数器的精确控制
        
        注意：这个函数在每个仿真步（0.0001s）都会被调用
        但只有达到更新间隔时，相应的控制器才会实际计算
        """
        current_step = self.step_count
        
        # ========== 1. 位置控制器更新 (50Hz) ==========
        if current_step % self.position_update_interval == 0:
            if 'position' in self.controllers:
                self._cached_target_velocity = self.controllers['position'].update(
                    self.target_position, drone_state.position
                )
                # print(f"Step {current_step}: 位置控制器更新 → 目标速度: {self._cached_target_velocity}")
        
        # ========== 2. 速度控制器更新 (50Hz) ==========
        if current_step % self.velocity_update_interval == 0:
            if 'velocity' in self.controllers:
                # 使用位置控制器的输出作为输入
                acceleration = self.controllers['velocity'].update(
                    self._cached_target_velocity, drone_state.velocity, self.sim_dt
                )
                
                # 计算推力向量
                self._cached_thrust_vector = self.controllers['velocity'].calculate_thrust_vector(acceleration)
                
                # 计算期望姿态（从推力向量和偏航角）
                if 'attitude' in self.controllers:
                    self._cached_desired_attitude = self.controllers['attitude'].calculate_desired_attitude(
                        self._cached_thrust_vector, self.target_yaw
                    )
                # print(f"Step {current_step}: 速度控制器更新 → 推力: {np.linalg.norm(self._cached_thrust_vector):.2f}N")
        
        # ========== 3. 姿态控制器更新 (250Hz) ==========
        if current_step % self.attitude_update_interval == 0:
            if 'attitude' in self.controllers:
                # 使用速度控制器的输出作为输入
                self._cached_target_angular_rate = self.controllers['attitude'].update(
                    self._cached_desired_attitude, drone_state.attitude
                )
                # print(f"Step {current_step}: 姿态控制器更新 → 目标角速度: {self._cached_target_angular_rate}")
        
        # ========== 4. 角速度控制器更新 (1000Hz) ==========
        if current_step % self.angle_rate_update_interval == 0:
            if 'angle_rate' in self.controllers:
                # 使用姿态控制器的输出作为输入
                self._cached_torque = self.controllers['angle_rate'].update(
                    self._cached_target_angular_rate, drone_state.angular_velocity, self.sim_dt
                )
                # print(f"Step {current_step}: 角速度控制器更新 → 扭矩: {self._cached_torque}")
        
        # ========== 5. 动力分配 (1000Hz) ==========
        # 每次角速度更新时都执行动力分配
        if current_step % self.angle_rate_update_interval == 0:
            thrust_magnitude = np.linalg.norm(self._cached_thrust_vector)
            motor_speeds = self.mixer.allocate(thrust_magnitude, self._cached_torque)
        else:
            # 使用上一次计算的电机转速
            motor_speeds = self.control_output.motor_speeds if hasattr(self, 'control_output') else np.zeros(4)
        
        # ========== 6. 构建输出 ==========
        self.control_output = ControlOutput(
            thrust_vector=self._cached_thrust_vector,
            torque=self._cached_torque,
            motor_speeds=motor_speeds,
            attitude_target=self._cached_desired_attitude,
            angular_rate_target=self._cached_target_angular_rate
        )
        
        # 步数增加
        self.step_count += 1
        
        return self.control_output
    
    def reset(self):
        """重置控制器状态"""
        super().reset()
        self.step_count = 0
        self._cached_target_velocity = np.zeros(3)
        self._cached_desired_attitude = quaternion.from_float_array([1, 0, 0, 0])
        self._cached_target_angular_rate = np.zeros(3)
        self._cached_thrust_vector = np.zeros(3)
        self._cached_torque = np.zeros(3)    