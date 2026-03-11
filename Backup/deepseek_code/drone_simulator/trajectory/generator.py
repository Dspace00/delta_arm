"""
轨迹生成器（从原代码迁移并扩展）
"""
import numpy as np
from typing import List, Tuple, Optional, Union
import math


def generate_circular_trajectory(
    center: List[float],
    radius: float,
    total_time: float,
    num_points: int = 50,
    start_angle: float = 0,
    clockwise: bool = False,
    height_variation: bool = False,
    height_amplitude: float = 0.5,
    hold_last: bool = True
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
        hold_last: 是否在最后保持位置
    
    返回:
        轨迹列表，每个元素为 (时间, [x, y, z])
    """
    trajectory = []
    direction = -1 if clockwise else 1
    
    # 计算时间步长
    if num_points > 1:
        time_step = total_time / (num_points - 1)
    else:
        time_step = total_time
    
    # 生成轨迹点
    for i in range(num_points):
        # 计算当前时间
        t = i * time_step
        
        # 计算当前角度
        if num_points > 1:
            angle = start_angle + direction * 2 * math.pi * i / (num_points - 1)
        else:
            angle = start_angle
        
        # 计算位置
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        
        # 计算Z坐标
        if height_variation:
            z = center[2] + height_amplitude * math.sin(angle * 2)
        else:
            z = center[2]
        
        trajectory.append((t, [x, y, z]))
    
    # 如果需要在最后保持位置
    if hold_last and trajectory:
        last_point = trajectory[-1]
        hold_time = last_point[0] + 1.0
        trajectory.append((hold_time, last_point[1]))
    
    return trajectory


def generate_spiral_trajectory(
    center: List[float],
    start_radius: float,
    end_radius: float,
    total_time: float,
    num_turns: int = 2,
    num_points: int = 100,
    clockwise: bool = False
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


def generate_line_trajectory(
    start_point: List[float],
    end_point: List[float],
    total_time: float,
    num_points: int = 50,
    hold_start: bool = True,
    hold_end: bool = True
) -> List[Tuple[float, List[float]]]:
    """
    生成直线轨迹
    
    参数:
        start_point: 起点 [x, y, z]
        end_point: 终点 [x, y, z]
        total_time: 总时间
        num_points: 点数
        hold_start: 是否在起点保持
        hold_end: 是否在终点保持
    
    返回:
        轨迹列表
    """
    trajectory = []
    
    if num_points < 2:
        num_points = 2
    
    # 计算时间步长
    time_step = total_time / (num_points - 1)
    
    # 生成轨迹点
    for i in range(num_points):
        t = i * time_step
        
        # 线性插值
        alpha = i / (num_points - 1) if num_points > 1 else 0
        position = [
            start_point[0] + alpha * (end_point[0] - start_point[0]),
            start_point[1] + alpha * (end_point[1] - start_point[1]),
            start_point[2] + alpha * (end_point[2] - start_point[2])
        ]
        
        trajectory.append((t, position))
    
    # 添加保持点
    if hold_start:
        trajectory.insert(0, (0.0, start_point))
    
    if hold_end:
        last_time = trajectory[-1][0]
        trajectory.append((last_time + 1.0, end_point))
    
    return trajectory


def generate_waypoint_trajectory(
    waypoints: List[Tuple[float, List[float]]],
    smooth: bool = True,
    velocity_profile: str = 'trapezoidal'  # 'trapezoidal', 's_curve'
) -> List[Tuple[float, List[float]]]:
    """
    生成通过一系列航点的轨迹
    
    参数:
        waypoints: 航点列表，每个元素为 (时间, [x, y, z])
        smooth: 是否平滑过渡
        velocity_profile: 速度曲线类型
    
    返回:
        插值后的轨迹
    """
    if len(waypoints) < 2:
        return waypoints
    
    trajectory = []
    
    # 对每段路径进行插值
    for i in range(len(waypoints) - 1):
        t_start, pos_start = waypoints[i]
        t_end, pos_end = waypoints[i + 1]
        
        # 生成该段的点数
        segment_time = t_end - t_start
        if segment_time <= 0:
            continue
        
        # 根据时间决定点数
        num_segment_points = max(2, int(segment_time * 10))  # 10Hz
        
        for j in range(num_segment_points):
            t = t_start + j * segment_time / (num_segment_points - 1)
            
            if smooth:
                # 平滑插值（使用sigmoid函数）
                alpha = j / (num_segment_points - 1)
                # sigmoid-like easing
                ease_alpha = alpha * alpha * (3 - 2 * alpha)
                
                position = [
                    pos_start[0] + ease_alpha * (pos_end[0] - pos_start[0]),
                    pos_start[1] + ease_alpha * (pos_end[1] - pos_start[1]),
                    pos_start[2] + ease_alpha * (pos_end[2] - pos_start[2])
                ]
            else:
                # 线性插值
                alpha = j / (num_segment_points - 1)
                position = [
                    pos_start[0] + alpha * (pos_end[0] - pos_start[0]),
                    pos_start[1] + alpha * (pos_end[1] - pos_start[1]),
                    pos_start[2] + alpha * (pos_end[2] - pos_start[2])
                ]
            
            trajectory.append((t, position))
    
    return trajectory


def calculate_trajectory_velocity(trajectory: List[Tuple[float, List[float]]]) -> List[Tuple[float, List[float]]]:
    """
    计算轨迹的速度剖面
    
    参数:
        trajectory: 位置轨迹
    
    返回:
        速度轨迹列表，每个元素为 (时间, [vx, vy, vz])
    """
    if len(trajectory) < 2:
        return [(trajectory[0][0], [0.0, 0.0, 0.0])]
    
    velocity_trajectory = []
    
    for i in range(len(trajectory)):
        t, position = trajectory[i]
        
        if i == 0:
            # 第一个点，使用前向差分
            t_next, pos_next = trajectory[i + 1]
            dt = t_next - t
            if dt > 0:
                velocity = [
                    (pos_next[0] - position[0]) / dt,
                    (pos_next[1] - position[1]) / dt,
                    (pos_next[2] - position[2]) / dt
                ]
            else:
                velocity = [0.0, 0.0, 0.0]
        
        elif i == len(trajectory) - 1:
            # 最后一个点，使用后向差分
            t_prev, pos_prev = trajectory[i - 1]
            dt = t - t_prev
            if dt > 0:
                velocity = [
                    (position[0] - pos_prev[0]) / dt,
                    (position[1] - pos_prev[1]) / dt,
                    (position[2] - pos_prev[2]) / dt
                ]
            else:
                velocity = [0.0, 0.0, 0.0]
        
        else:
            # 中间点，使用中心差分
            t_prev, pos_prev = trajectory[i - 1]
            t_next, pos_next = trajectory[i + 1]
            dt = t_next - t_prev
            if dt > 0:
                velocity = [
                    (pos_next[0] - pos_prev[0]) / dt,
                    (pos_next[1] - pos_prev[1]) / dt,
                    (pos_next[2] - pos_prev[2]) / dt
                ]
            else:
                velocity = [0.0, 0.0, 0.0]
        
        velocity_trajectory.append((t, velocity))
    
    return velocity_trajectory