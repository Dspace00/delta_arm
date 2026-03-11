# DeltaController.py
import numpy as np
import time

class DeltaController:
    """Delta机械臂控制器"""
    
    def __init__(self, robot_model):
        self.robot_model = robot_model
        
        # 控制参数
        self.max_joint_speed = 2.0    # 最大关节速度 (rad/s)
        self.max_joint_accel = 5.0    # 最大关节加速度 (rad/s²)
        
        # 轨迹规划
        self.trajectories = []
        self.is_moving = False
        self.move_start_time = 0
        self.move_duration = 0
        self.target_angles = None
        
    def plan_trapezoidal_trajectory(self, start, target, max_speed, max_accel):
        """规划梯形速度曲线轨迹"""
        distance = abs(target - start)
        direction = 1 if target > start else -1
        
        # 计算达到最大速度所需的时间和距离
        t_accel = max_speed / max_accel
        s_accel = 0.5 * max_accel * t_accel**2
        
        if 2 * s_accel <= distance:
            # 梯形曲线：加速-匀速-减速
            t_cruise = (distance - 2 * s_accel) / max_speed
            duration = 2 * t_accel + t_cruise
            return {
                'start': start,
                'target': target,
                'max_speed': max_speed,
                'max_accel': max_accel,
                't_accel': t_accel,
                't_cruise': t_cruise,
                'duration': duration,
                'direction': direction
            }
        else:
            # 三角曲线：加速后立即减速
            max_speed_actual = np.sqrt(max_accel * distance)
            t_accel = max_speed_actual / max_accel
            duration = 2 * t_accel
            return {
                'start': start,
                'target': target,
                'max_speed': max_speed_actual,
                'max_accel': max_accel,
                't_accel': t_accel,
                't_cruise': 0,
                'duration': duration,
                'direction': direction
            }
    
    def get_trapezoidal_position(self, traj, t):
        """根据梯形速度曲线获取当前位置"""
        if t <= traj['t_accel']:
            # 加速阶段
            s = 0.5 * traj['max_accel'] * t**2
        elif t <= traj['t_accel'] + traj['t_cruise']:
            # 匀速阶段
            s_accel = 0.5 * traj['max_accel'] * traj['t_accel']**2
            s_cruise = traj['max_speed'] * (t - traj['t_accel'])
            s = s_accel + s_cruise
        else:
            # 减速阶段
            t_decel = t - (traj['t_accel'] + traj['t_cruise'])
            s_accel = 0.5 * traj['max_accel'] * traj['t_accel']**2
            s_cruise = traj['max_speed'] * traj['t_cruise']
            s_decel = traj['max_speed'] * t_decel - 0.5 * traj['max_accel'] * t_decel**2
            s = s_accel + s_cruise + s_decel
        
        return traj['start'] + traj['direction'] * s
    
    def move_to_position(self, target_pos, current_angles, speed=1.0):
        """
        移动到目标位置
        返回: 轨迹规划结果
        """
        # 计算逆运动学
        target_angles = self.robot_model.inverse_kinematics(target_pos)
        if target_angles is None:
            return None
        
        # 将速度参数转换为实际速度
        actual_speed = speed * self.max_joint_speed
        actual_accel = speed * self.max_joint_accel
        
        # 为每个关节规划轨迹
        trajectories = []
        total_duration = 0
        
        for i in range(3):
            traj = self.plan_trapezoidal_trajectory(
                current_angles[i], target_angles[i], 
                actual_speed, actual_accel
            )
            trajectories.append(traj)
            total_duration = max(total_duration, traj['duration'])
        
        return {
            'trajectories': trajectories,
            'target_angles': target_angles,
            'duration': total_duration,
            'target_position': target_pos
        }
    
    def get_current_target_angles(self, trajectories, elapsed_time):
        """获取当前时刻的目标角度"""
        if elapsed_time >= trajectories['duration']:
            return trajectories['target_angles']
        
        current_angles = []
        for traj in trajectories['trajectories']:
            angle = self.get_trapezoidal_position(traj, elapsed_time)
            current_angles.append(angle)
        
        return current_angles
    
    def trajectory_circle(self, center, radius, speed=1.0, duration=10.0, dt=0.01):
        """生成圆形轨迹点"""
        points = []
        n_points = int(duration / dt)
        
        for i in range(n_points):
            t = i * dt
            angle = 2 * np.pi * t / duration
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            
            points.append([x, y, z])
        
        return points
    
    def trajectory_square(self, center, side_length, speed=1.0, duration=10.0, dt=0.01):
        """生成正方形轨迹点"""
        points = []
        half_side = side_length / 2
        corners = [
            [center[0] - half_side, center[1] - half_side, center[2]],
            [center[0] + half_side, center[1] - half_side, center[2]],
            [center[0] + half_side, center[1] + half_side, center[2]],
            [center[0] - half_side, center[1] + half_side, center[2]]
        ]
        
        side_time = duration / 4
        points_per_side = int(side_time / dt)
        
        for i in range(4):
            start_corner = corners[i]
            end_corner = corners[(i + 1) % 4]
            
            for j in range(points_per_side):
                t = j / points_per_side
                x = start_corner[0] + (end_corner[0] - start_corner[0]) * t
                y = start_corner[1] + (end_corner[1] - start_corner[1]) * t
                z = start_corner[2] + (end_corner[2] - start_corner[2]) * t
                
                points.append([x, y, z])
        
        return points
    
    def trajectory_figure8(self, center, size, tilt_angle=0, speed=1.0, duration=10.0, dt=0.01):
        """生成8字形轨迹点"""
        points = []
        n_points = int(duration / dt)
        tilt_rad = np.radians(tilt_angle)
        
        for i in range(n_points):
            t = i * dt
            angle = 4 * np.pi * t / duration  # 两个完整的圆
            
            # 8字轨迹参数方程
            u = size * np.sin(angle)
            v = size * np.sin(angle) * np.cos(angle)
            
            # 应用倾斜变换
            x = center[0] + u
            y = center[1] + v * np.cos(tilt_rad)
            z = center[2] + v * np.sin(tilt_rad)
            
            points.append([x, y, z])
        
        return points