#!/usr/bin/env python3
"""
简化的仿真测试（使用新控制器）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import quaternion
import time
from drone_simulator.utils import ConfigManager
from drone_simulator.controllers import PIDControllerFactory
from drone_simulator.models.mixer import Mixer


class SimpleDroneSimulation:
    """简化的无人机仿真（用于测试新控制器）"""
    
    def __init__(self, config_path=None):
        """初始化"""
        # 加载配置
        self.config_manager = ConfigManager()
        if config_path and os.path.exists(config_path):
            self.config_manager.load_from_files(
                f"{config_path}_drone.yaml" if os.path.exists(f"{config_path}_drone.yaml") else None,
                f"{config_path}_controller.yaml" if os.path.exists(f"{config_path}_controller.yaml") else None,
                f"{config_path}_simulation.yaml" if os.path.exists(f"{config_path}_simulation.yaml") else None
            )
        else:
            self.config_manager.load_default()
        
        # 无人机参数
        self.mass = self.config_manager.drone.mass
        self.inertia = np.array([self.config_manager.drone.inertia['Ixx'],
                                self.config_manager.drone.inertia['Iyy'],
                                self.config_manager.drone.inertia['Izz']])
        
        # 创建控制器
        self.controllers = PIDControllerFactory.create_from_controller_config(
            self.config_manager.controller,
            self.mass,
            self.inertia
        )
        
        # 创建混控器
        self.mixer = Mixer(self.config_manager.drone.motors)
        
        # 初始状态
        self.position = np.array(self.config_manager.simulation.initial_conditions['position'])
        self.velocity = np.array(self.config_manager.simulation.initial_conditions['velocity'])
        self.attitude = quaternion.from_float_array(
            self.config_manager.simulation.initial_conditions['attitude']
        )
        self.angular_rate = np.array(self.config_manager.simulation.initial_conditions['angular_velocity'])
        
        # 目标状态
        self.target_position = self.position.copy()
        self.target_yaw = 0.0
        
        # 控制命令
        self.thrust_vector = np.zeros(3)
        self.torque_command = np.zeros(3)
        self.motor_speeds = np.zeros(4)
        
        # 数据记录
        self.history = {
            'time': [],
            'position': [],
            'target_position': [],
            'velocity': [],
            'attitude': [],
            'thrust': [],
            'torque': []
        }
        
        print(f"Initialized SimpleDroneSimulation with {len(self.controllers)} controllers")
    
    def set_target(self, position, yaw=0.0):
        """设置目标"""
        self.target_position = np.array(position)
        self.target_yaw = yaw
    
    def update(self, dt):
        """更新仿真一步"""
        # 位置控制
        if 'position' in self.controllers:
            target_velocity = self.controllers['position'].update(
                self.target_position, self.position
            )
        else:
            target_velocity = np.zeros(3)
        
        # 速度控制
        if 'velocity' in self.controllers:
            acceleration = self.controllers['velocity'].update(
                target_velocity, self.velocity, dt
            )
            self.thrust_vector = self.controllers['velocity'].calculate_thrust_vector(acceleration)
        else:
            self.thrust_vector = np.array([0, 0, self.mass * 9.81])
        
        # 姿态控制
        if 'attitude' in self.controllers:
            # 计算期望姿态
            target_attitude = self.controllers['attitude'].calculate_desired_attitude(
                self.thrust_vector, self.target_yaw
            )
            
            # 姿态控制
            target_angular_rate = self.controllers['attitude'].update(
                target_attitude, self.attitude
            )
        else:
            target_angular_rate = np.zeros(3)
        
        # 角速度控制
        if 'angle_rate' in self.controllers:
            self.torque_command = self.controllers['angle_rate'].update(
                target_angular_rate, self.angular_rate, dt
            )
        else:
            self.torque_command = np.zeros(3)
        
        # 动力分配
        thrust_magnitude = np.linalg.norm(self.thrust_vector)
        self.motor_speeds = self.mixer.allocate(thrust_magnitude, self.torque_command)
        
        # 简单物理更新（简化模型）
        # 位置更新
        self.position += self.velocity * dt
        
        # 速度更新（假设推力立即生效）
        acceleration_total = self.thrust_vector / self.mass + np.array([0, 0, -9.81])
        self.velocity += acceleration_total * dt
        
        # 姿态更新（简化）
        # 在实际仿真中，这里应该使用四元数积分
        
        return self.position, self.velocity, self.attitude
    
    def run(self, duration=5.0, dt=0.02):
        """运行仿真"""
        print(f"\nRunning simulation for {duration} seconds (dt={dt})")
        
        start_time = time.time()
        sim_time = 0.0
        step = 0
        
        # 设置目标轨迹
        targets = [
            (0.0, [0, 0, 1.0]),
            (2.0, [2, 0, 1.5]),
            (4.0, [2, 2, 2.0]),
            (6.0, [0, 2, 1.5]),
            (8.0, [0, 0, 1.0])
        ]
        
        while sim_time < duration:
            # 更新目标
            for i in range(len(targets) - 1):
                t_start, pos_start = targets[i]
                t_end, pos_end = targets[i + 1]
                
                if t_start <= sim_time < t_end:
                    alpha = (sim_time - t_start) / (t_end - t_start)
                    target_pos = np.array(pos_start) + alpha * (np.array(pos_end) - np.array(pos_start))
                    self.set_target(target_pos)
                    break
            
            # 更新仿真
            position, velocity, attitude = self.update(dt)
            
            # 记录数据
            self.history['time'].append(sim_time)
            self.history['position'].append(position.copy())
            self.history['target_position'].append(self.target_position.copy())
            self.history['velocity'].append(velocity.copy())
            self.history['attitude'].append(attitude.copy())
            self.history['thrust'].append(self.thrust_vector.copy())
            self.history['torque'].append(self.torque_command.copy())
            
            sim_time += dt
            step += 1
            
            # 显示进度
            if step % 50 == 0:
                elapsed = time.time() - start_time
                progress = sim_time / duration * 100
                print(f"Progress: {progress:.1f}%, Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
        
        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.2f} seconds")
        print(f"Real-time factor: {duration / elapsed:.2f}x")
        
        return self.history
    
    def plot_results(self):
        """绘制结果（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            time_array = np.array(self.history['time'])
            position = np.array(self.history['position'])
            target_position = np.array(self.history['target_position'])
            velocity = np.array(self.history['velocity'])
            thrust = np.array(self.history['thrust'])
            torque = np.array(self.history['torque'])
            
            # 位置跟踪
            axes[0, 0].plot(time_array, position[:, 0], 'r-', label='X')
            axes[0, 0].plot(time_array, position[:, 1], 'g-', label='Y')
            axes[0, 0].plot(time_array, position[:, 2], 'b-', label='Z')
            axes[0, 0].plot(time_array, target_position[:, 0], 'r--', label='Target X')
            axes[0, 0].plot(time_array, target_position[:, 1], 'g--', label='Target Y')
            axes[0, 0].plot(time_array, target_position[:, 2], 'b--', label='Target Z')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].set_title('Position Tracking')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 速度
            axes[0, 1].plot(time_array, velocity[:, 0], 'r-', label='Vx')
            axes[0, 1].plot(time_array, velocity[:, 1], 'g-', label='Vy')
            axes[0, 1].plot(time_array, velocity[:, 2], 'b-', label='Vz')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            axes[0, 1].set_title('Velocity')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 推力
            axes[0, 2].plot(time_array, thrust[:, 0], 'r-', label='Fx')
            axes[0, 2].plot(time_array, thrust[:, 1], 'g-', label='Fy')
            axes[0, 2].plot(time_array, thrust[:, 2], 'b-', label='Fz')
            axes[0, 2].set_xlabel('Time (s)')
            axes[0, 2].set_ylabel('Thrust (N)')
            axes[0, 2].set_title('Thrust Vector')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
            
            # 扭矩
            axes[1, 0].plot(time_array, torque[:, 0], 'r-', label='Mx')
            axes[1, 0].plot(time_array, torque[:, 1], 'g-', label='My')
            axes[1, 0].plot(time_array, torque[:, 2], 'b-', label='Mz')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Torque (Nm)')
            axes[1, 0].set_title('Torque Command')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 3D轨迹
            ax_3d = fig.add_subplot(2, 3, 5, projection='3d')
            ax_3d.plot(position[:, 0], position[:, 1], position[:, 2], 'b-', label='Trajectory')
            ax_3d.plot(target_position[:, 0], target_position[:, 1], target_position[:, 2], 'r--', label='Target')
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title('3D Trajectory')
            ax_3d.legend()
            ax_3d.grid(True)
            
            # 位置误差
            position_error = np.linalg.norm(position - target_position, axis=1)
            axes[1, 2].plot(time_array, position_error, 'k-')
            axes[1, 2].set_xlabel('Time (s)')
            axes[1, 2].set_ylabel('Error (m)')
            axes[1, 2].set_title('Position Error')
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not installed. Skipping plots.")
            # 打印统计数据
            position_error = np.linalg.norm(
                np.array(self.history['position']) - np.array(self.history['target_position']),
                axis=1
            )
            print(f"Average position error: {np.mean(position_error):.3f} m")
            print(f"Max position error: {np.max(position_error):.3f} m")


if __name__ == "__main__":
    print("Simple Drone Simulation with New Controllers\n")
    
    # 创建仿真
    sim = SimpleDroneSimulation()
    
    # 运行仿真
    history = sim.run(duration=8.0, dt=0.02)
    
    # 绘制结果
    sim.plot_results()