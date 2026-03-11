import numpy as np
import math
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


class TrajectoryPlanner:
    def __init__(self, trajectory_points: List[Tuple[float, List[float]]]):
        """
        初始化轨迹规划器
        
        参数:
            trajectory_points: 轨迹点列表 [(time, [x, y, z]), ...]
        """
        self.trajectory = trajectory_points
        # 确保按时间排序
        self.trajectory.sort(key=lambda x: x[0])

    def get_target_position(self, current_time: float) -> np.ndarray:
        """
        根据当前时间获取目标位置（线性插值）
        """
        if not self.trajectory:
             return np.zeros(3)
             
        # 如果时间小于起点，返回起点位置
        if current_time <= self.trajectory[0][0]:
            return np.array(self.trajectory[0][1])
            
        # 如果时间大于终点，返回终点位置
        if current_time >= self.trajectory[-1][0]:
            return np.array(self.trajectory[-1][1])
            
        # 寻找对应的时间段并插值
        for i in range(len(self.trajectory) - 1):
            t_start, pos_start = self.trajectory[i]
            t_end, pos_end = self.trajectory[i + 1]
            
            if t_start <= current_time < t_end:
                alpha = (current_time - t_start) / (t_end - t_start)
                target_pos = np.array(pos_start) + alpha * (np.array(pos_end) - np.array(pos_start))
                return target_pos
                
        return np.array(self.trajectory[-1][1])
