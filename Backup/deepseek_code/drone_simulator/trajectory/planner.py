"""
轨迹规划器
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from .generator import *


class TrajectoryPlanner:
    """轨迹规划器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化轨迹规划器
        
        Args:
            config: 规划器配置
        """
        self.config = config or {}
        
        # 默认参数
        self.default_params = {
            'max_velocity': 2.0,      # m/s
            'max_acceleration': 1.0,  # m/s²
            'max_jerk': 2.0,          # m/s³
            'safety_margin': 0.5,     # m
            'sampling_rate': 50       # Hz
        }
        
        # 更新配置
        if config:
            self.default_params.update(config)
        
        # 当前轨迹
        self.current_trajectory = []
        self.current_index = 0
        self.is_complete = True
        
    def plan_circle(self, center: List[float], radius: float, 
                   total_time: float, **kwargs) -> List[Tuple[float, List[float]]]:
        """
        规划圆形轨迹
        
        Returns:
            时间-位置轨迹
        """
        trajectory = generate_circular_trajectory(center, radius, total_time, **kwargs)
        self.current_trajectory = trajectory
        self.current_index = 0
        self.is_complete = False
        
        return trajectory
    
    def plan_spiral(self, center: List[float], start_radius: float, 
                   end_radius: float, total_time: float, **kwargs) -> List[Tuple[float, List[float]]]:
        """
        规划螺旋轨迹
        """
        trajectory = generate_spiral_trajectory(
            center, start_radius, end_radius, total_time, **kwargs
        )
        self.current_trajectory = trajectory
        self.current_index = 0
        self.is_complete = False
        
        return trajectory
    
    def plan_line(self, start: List[float], end: List[float], 
                 total_time: float, **kwargs) -> List[Tuple[float, List[float]]]:
        """
        规划直线轨迹
        """
        trajectory = generate_line_trajectory(start, end, total_time, **kwargs)
        self.current_trajectory = trajectory
        self.current_index = 0
        self.is_complete = False
        
        return trajectory
    
    def plan_waypoints(self, waypoints: List[Tuple[float, List[float]]], 
                      **kwargs) -> List[Tuple[float, List[float]]]:
        """
        规划航点轨迹
        """
        trajectory = generate_waypoint_trajectory(waypoints, **kwargs)
        self.current_trajectory = trajectory
        self.current_index = 0
        self.is_complete = False
        
        return trajectory
    
    def get_current_target(self, current_time: float) -> Tuple[Optional[List[float]], bool]:
        """
        获取当前时间的目标位置
        
        Args:
            current_time: 当前时间
            
        Returns:
            (目标位置, 是否完成)
        """
        if self.is_complete or not self.current_trajectory:
            return None, True
        
        # 寻找当前时间对应的轨迹点
        while (self.current_index < len(self.current_trajectory) - 1 and 
               self.current_trajectory[self.current_index + 1][0] <= current_time):
            self.current_index += 1
        
        # 检查是否到达终点
        if self.current_index >= len(self.current_trajectory) - 1:
            if current_time >= self.current_trajectory[-1][0]:
                self.is_complete = True
                return self.current_trajectory[-1][1], True
        
        # 线性插值
        if self.current_index < len(self.current_trajectory) - 1:
            t1, pos1 = self.current_trajectory[self.current_index]
            t2, pos2 = self.current_trajectory[self.current_index + 1]
            
            if t2 > t1:
                alpha = (current_time - t1) / (t2 - t1)
                alpha = np.clip(alpha, 0.0, 1.0)
                
                target_pos = [
                    pos1[0] + alpha * (pos2[0] - pos1[0]),
                    pos1[1] + alpha * (pos2[1] - pos1[1]),
                    pos1[2] + alpha * (pos2[2] - pos1[2])
                ]
                
                return target_pos, False
        
        # 返回当前点的位置
        return self.current_trajectory[self.current_index][1], self.is_complete
    
    def reset(self):
        """重置规划器"""
        self.current_trajectory = []
        self.current_index = 0
        self.is_complete = True
    
    def is_active(self) -> bool:
        """检查是否有激活的轨迹"""
        return not self.is_complete and len(self.current_trajectory) > 0
    
    def get_progress(self) -> float:
        """获取轨迹执行进度"""
        if not self.current_trajectory or self.is_complete:
            return 1.0
        
        if len(self.current_trajectory) == 1:
            return 1.0
        
        total_time = self.current_trajectory[-1][0] - self.current_trajectory[0][0]
        if total_time <= 0:
            return 1.0
        
        current_time = self.current_trajectory[self.current_index][0]
        return min(current_time / total_time, 1.0)
    
    def __str__(self):
        status = "Active" if self.is_active() else "Inactive"
        progress = f"{self.get_progress()*100:.1f}%" if self.is_active() else "N/A"
        return f"TrajectoryPlanner(status={status}, progress={progress}, points={len(self.current_trajectory)})"