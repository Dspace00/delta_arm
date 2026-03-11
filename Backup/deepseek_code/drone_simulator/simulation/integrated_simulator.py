"""
集成的无人机仿真系统
"""
import mujoco
import mujoco.viewer
import numpy as np
import quaternion
from typing import Optional, Dict, Any, List, Tuple
import time
from scipy.spatial.transform import Rotation as R

from ..controllers.cascaded_controller import CascadedController
from ..trajectory import TrajectoryPlanner
from ..models.drone import DroneState, DroneModel
from ..utils import get_global_config


class IntegratedDroneSimulator:
    """集成的无人机仿真器"""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        初始化仿真器
        
        Args:
            model_path: MuJoCo模型文件路径
            config_path: 配置文件路径
        """
        # 加载配置
        if config_path:
            self.config = get_global_config(config_path)
        else:
            self.config = get_global_config()
        
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 仿真参数
        self.sim_dt = self.model.opt.timestep
        self.sim_time = 0.0
        self.step_count = 0
        
        # 无人机模型
        self.drone_model = DroneModel(
            mass=self.config.drone.mass,
            arm_length=self.config.drone.dimensions['arm_length']
        )
        
        # 创建控制器
        self.controller = CascadedController(self.config)
        self.controller.set_simulation_timestep(self.sim_dt)
        # 创建轨迹规划器
        self.trajectory_planner = TrajectoryPlanner()
        
        # 可视化
        self.viewer = None
        self.render_enabled = self.config.simulation.visualization['enabled']
        self.target_fps = self.config.simulation.visualization['fps']
        self.render_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        self.last_render_time = 0.0
        
        # 数据记录
        self.logging_enabled = self.config.simulation.logging['enabled']
        self.log_data = []
        
        # 状态变量
        self.is_running = False
        
        # 打印信息
        print(f"=== Integrated Drone Simulator ===")
        print(f"Model: {model_path}")
        print(f"Simulation dt: {self.sim_dt:.4f} s")
        print(f"Mass: {self.drone_model.mass:.2f} kg")
        print(f"Control mode: {self.controller.control_mode}")
        print(f"Render enabled: {self.render_enabled}")
        print(f"Logging enabled: {self.logging_enabled}")
    
    def get_drone_state(self) -> DroneState:
        """从MuJoCo获取无人机状态"""
        # 获取无人机主体ID
        body_name = self.config.drone.body_name
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        if body_id == -1:
            # 使用第一个主体
            body_id = 0
        
        # 位置和四元数
        position = self.data.body(body_id).xpos.copy()
        quat = self.data.body(body_id).xquat.copy()  # [w, x, y, z]
        
        # 转换四元数顺序
        attitude = quaternion.from_float_array([quat[0], quat[1], quat[2], quat[3]])
        
        # 线速度和角速度
        velocity = self.data.body(body_id).cvel[3:6].copy()  # 线性速度
        
        # 角速度（世界坐标系转换到机体坐标系）
        angular_velocity_world = self.data.body(body_id).cvel[0:3].copy()
        attitude_matrix = self.data.body(body_id).xmat.copy().reshape(3, 3)
        angular_velocity_body = attitude_matrix.T @ angular_velocity_world
        
        # 创建状态对象
        state = DroneState(
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=angular_velocity_body,
            timestamp=self.sim_time
        )
        
        return state
    
    def set_initial_state(self, position: Optional[List[float]] = None,
                         attitude: Optional[List[float]] = None):
        """设置初始状态"""
        # 重置MuJoCo数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置位置
        if position is not None:
            body_name = self.config.drone.body_name
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                self.data.body(body_id).xpos[:] = position
        
        # 设置姿态
        if attitude is not None:
            body_name = self.config.drone.body_name
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                self.data.body(body_id).xquat[:] = attitude
        
        # 重置控制器
        self.controller.reset()
        
        # 重置时间
        self.sim_time = 0.0
        self.step_count = 0
        
        print("Initial state set")
    
    def apply_control(self, control_output):
        """应用控制到MuJoCo执行器"""
        # 将控制输出应用到执行器
        # 注意：这里需要根据实际模型调整
        
        # 示例：使用虚拟力/扭矩执行器
        body_name = self.config.drone.body_name
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        if body_id != -1:
            # 获取机体姿态矩阵
            attitude_matrix = self.data.body(body_id).xmat.copy().reshape(3, 3)
            
            # 将推力从机体坐标系转换到世界坐标系
            thrust_body = control_output.thrust_vector
            thrust_world = attitude_matrix @ thrust_body
            
            # 应用力（在MuJoCo中，力是在世界坐标系中施加的）
            # 这里简化处理，实际可能需要使用执行器
            
            # 如果有力执行器，设置控制值
            for i in range(3):
                actuator_name = f"force{i+1}"
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id != -1:
                    self.data.ctrl[actuator_id] = thrust_world[i]
            
            # 应用扭矩
            for i in range(3):
                actuator_name = f"torque{i+1}"
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id != -1:
                    self.data.ctrl[actuator_id] = control_output.torque[i]
    
    def plan_trajectory(self, trajectory_type: str, **kwargs):
        """规划轨迹"""
        if trajectory_type == 'circle':
            trajectory = self.trajectory_planner.plan_circle(**kwargs)
        elif trajectory_type == 'spiral':
            trajectory = self.trajectory_planner.plan_spiral(**kwargs)
        elif trajectory_type == 'line':
            trajectory = self.trajectory_planner.plan_line(**kwargs)
        elif trajectory_type == 'waypoints':
            trajectory = self.trajectory_planner.plan_waypoints(**kwargs)
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        
        print(f"Planned {trajectory_type} trajectory with {len(trajectory)} points")
        return trajectory
    
    def update_trajectory_target(self):
        """根据轨迹规划器更新目标位置"""
        if self.trajectory_planner.is_active():
            target_pos, is_complete = self.trajectory_planner.get_current_target(self.sim_time)
            if target_pos is not None:
                self.controller.set_target(position=target_pos)
            
            if is_complete:
                print("Trajectory completed")
    
    def step(self):
        """执行一个仿真步"""
        # 获取当前状态
        drone_state = self.get_drone_state()
        
        # 更新无人机模型状态
        self.drone_model.update_state(drone_state)
        
        # 更新轨迹目标
        self.update_trajectory_target()
        
        # 更新控制器
        control_output = self.controller.update(drone_state)
        
        # 应用控制
        self.apply_control(control_output)
        
        # 执行仿真步
        mujoco.mj_step(self.model, self.data)
        
        # 更新时间和计数器
        self.sim_time += self.sim_dt
        self.step_count += 1
        
        # 记录数据
        if self.logging_enabled and self.step_count % 10 == 0:
            self._log_data(drone_state, control_output)
        
        return drone_state, control_output
# -----------------
# 
    
    def step(self):
        """执行一个仿真步（核心循环）"""
        # 获取当前状态
        drone_state = self.get_drone_state()
        drone_state.timestamp = self.sim_time
        
        # 更新控制器（传入当前状态）
        # 注意：控制器内部使用步数计数器，不依赖仿真时间
        control_output = self.controller.update(drone_state)
        
        # 应用控制到执行器
        self.apply_control(control_output)
        
        # 执行MuJoCo物理仿真步
        mujoco.mj_step(self.model, self.data)
        
        # 更新仿真时间和步数
        self.sim_time += self.sim_dt
        self.step_count += 1
        
        # 记录数据
        if self.logging_enabled and self.step_count % 100 == 0:  # 每100步记录一次
            self._log_data(drone_state, control_output)
        
        return drone_state, control_output

# -----------------
    def _log_data(self, drone_state: DroneState, control_output):
        """记录数据"""
        log_entry = {
            'time': self.sim_time,
            'position': drone_state.position.copy(),
            'velocity': drone_state.velocity.copy(),
            'attitude': quaternion.as_float_array(drone_state.attitude).copy(),
            'angular_velocity': drone_state.angular_velocity.copy(),
            'target_position': self.controller.target_position.copy(),
            'target_attitude': quaternion.as_float_array(self.controller.target_attitude).copy(),
            'thrust_vector': control_output.thrust_vector.copy(),
            'torque': control_output.torque.copy(),
            'motor_speeds': control_output.motor_speeds.copy()
        }
        
        self.log_data.append(log_entry)
    
    def launch_viewer(self):
        """启动MuJoCo查看器"""
        if self.render_enabled:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("MuJoCo viewer launched")
    
    def update_viewer(self):
        """更新查看器显示"""
        if self.render_enabled and self.viewer is not None:
            current_time = time.time()
            if current_time - self.last_render_time >= self.render_interval:
                self.viewer.sync()
                self.last_render_time = current_time
    
    def close_viewer(self):
        """关闭查看器"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def run(self, duration: float, real_time: bool = False, 
           trajectory: Optional[List[Tuple[float, List[float]]]] = None):
        """
        运行仿真
        
        Args:
            duration: 仿真持续时间（秒）
            real_time: 是否实时运行
            trajectory: 轨迹，如果不为None则使用轨迹
        """
        # 设置轨迹
        if trajectory is not None:
            self.trajectory_planner.plan_waypoints(trajectory)
        
        # 计算总步数
        total_steps = int(duration / self.sim_dt)
        
        # 启动查看器
        if self.render_enabled:
            self.launch_viewer()
        
        print(f"\nStarting simulation for {duration:.1f} seconds ({total_steps} steps)")
        print(f"Real-time mode: {real_time}")
        
        start_time = time.time()
        self.is_running = True
        
        try:
            for step in range(total_steps):
                # 执行仿真步
                drone_state, control_output = self.step()
                
                # 更新查看器
                self.update_viewer()
                
                # 实时控制
                # if real_time:
                #     elapsed = time.time() - start_time
                #     target_time = (step + 1) * self.sim_dt
                #     if elapsed < target_time:
                #         time.sleep(target_time - elapsed)
                
                # 显示进度
                if step % 1000 == 0:
                    progress = (step + 1) / total_steps * 100
                    pos = drone_state.position
                    print(f"Progress: {progress:.1f}%, Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                
                # 检查是否完成
                if not self.is_running:
                    break
        
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        
        finally:
            self.is_running = False
            self.close_viewer()
            
            # 计算统计信息
            elapsed = time.time() - start_time
            real_time_factor = duration / elapsed if elapsed > 0 else 0
            
            print(f"\n=== Simulation Complete ===")
            print(f"Total steps: {self.step_count}")
            print(f"Actual time: {elapsed:.2f} s")
            print(f"Real-time factor: {real_time_factor:.2f}x")
            print(f"Final position: {drone_state.position}")
            
            # 保存数据
            if self.logging_enabled:
                self.save_log_data()
    
    def save_log_data(self, filename: str = None):
        """保存日志数据"""
        if not self.log_data:
            print("No data to save")
            return
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_log_{timestamp}.npz"
        
        # 提取数据
        times = np.array([entry['time'] for entry in self.log_data])
        positions = np.array([entry['position'] for entry in self.log_data])
        velocities = np.array([entry['velocity'] for entry in self.log_data])
        attitudes = np.array([entry['attitude'] for entry in self.log_data])
        angular_velocities = np.array([entry['angular_velocity'] for entry in self.log_data])
        target_positions = np.array([entry['target_position'] for entry in self.log_data])
        target_attitudes = np.array([entry['target_attitude'] for entry in self.log_data])
        thrust_vectors = np.array([entry['thrust_vector'] for entry in self.log_data])
        torques = np.array([entry['torque'] for entry in self.log_data])
        motor_speeds = np.array([entry['motor_speeds'] for entry in self.log_data])
        
        np.savez_compressed(
            filename,
            time=times,
            position=positions,
            velocity=velocities,
            attitude=attitudes,
            angular_velocity=angular_velocities,
            target_position=target_positions,
            target_attitude=target_attitudes,
            thrust_vector=thrust_vectors,
            torque=torques,
            motor_speeds=motor_speeds
        )
        
        print(f"Log data saved to {filename}")
    
    def plot_results(self):
        """绘制结果"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.log_data:
                print("No data to plot")
                return
            
            # 提取数据
            times = np.array([entry['time'] for entry in self.log_data])
            positions = np.array([entry['position'] for entry in self.log_data])
            target_positions = np.array([entry['target_position'] for entry in self.log_data])
            velocities = np.array([entry['velocity'] for entry in self.log_data])
            thrust_vectors = np.array([entry['thrust_vector'] for entry in self.log_data])
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 位置跟踪
            axes[0, 0].plot(times, positions[:, 0], 'r-', label='X')
            axes[0, 0].plot(times, positions[:, 1], 'g-', label='Y')
            axes[0, 0].plot(times, positions[:, 2], 'b-', label='Z')
            axes[0, 0].plot(times, target_positions[:, 0], 'r--', label='Target X')
            axes[0, 0].plot(times, target_positions[:, 1], 'g--', label='Target Y')
            axes[0, 0].plot(times, target_positions[:, 2], 'b--', label='Target Z')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].set_title('Position Tracking')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 速度
            axes[0, 1].plot(times, velocities[:, 0], 'r-', label='Vx')
            axes[0, 1].plot(times, velocities[:, 1], 'g-', label='Vy')
            axes[0, 1].plot(times, velocities[:, 2], 'b-', label='Vz')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            axes[0, 1].set_title('Velocity')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 推力
            axes[0, 2].plot(times, thrust_vectors[:, 0], 'r-', label='Fx')
            axes[0, 2].plot(times, thrust_vectors[:, 1], 'g-', label='Fy')
            axes[0, 2].plot(times, thrust_vectors[:, 2], 'b-', label='Fz')
            axes[0, 2].set_xlabel('Time (s)')
            axes[0, 2].set_ylabel('Thrust (N)')
            axes[0, 2].set_title('Thrust Vector')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
            
            # 位置误差
            position_error = np.linalg.norm(positions - target_positions, axis=1)
            axes[1, 0].plot(times, position_error, 'k-')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Error (m)')
            axes[1, 0].set_title('Position Error')
            axes[1, 0].grid(True)
            
            # 3D轨迹
            ax_3d = fig.add_subplot(2, 3, 5, projection='3d')
            ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Actual')
            ax_3d.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 'r--', label='Target')
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title('3D Trajectory')
            ax_3d.legend()
            ax_3d.grid(True)
            
            # 推力大小
            thrust_magnitude = np.linalg.norm(thrust_vectors, axis=1)
            axes[1, 2].plot(times, thrust_magnitude, 'm-')
            axes[1, 2].set_xlabel('Time (s)')
            axes[1, 2].set_ylabel('Thrust (N)')
            axes[1, 2].set_title('Thrust Magnitude')
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not installed. Skipping plots.")