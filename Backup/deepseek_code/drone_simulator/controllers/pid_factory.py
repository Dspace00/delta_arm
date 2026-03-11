"""
PID控制器工厂
用于根据配置创建控制器
"""
import numpy as np

from typing import Dict, Any, Optional
from .position_controller import PositionController
from .velocity_controller import VelocityController
from .attitude_controller import AttitudeController
from .angular_rate_controller import AngularRateController
from ..config.control_config import ControllerConfig


class PIDControllerFactory:
    """PID控制器工厂"""
    
    @staticmethod
    def create_position_controller(config: Dict[str, Any], **kwargs):
        """创建位置控制器"""
        return PositionController(config, **kwargs)
    
    @staticmethod
    def create_velocity_controller(config: Dict[str, Any], mass: float, **kwargs):
        """创建速度控制器"""
        return VelocityController(config, mass=mass, **kwargs)
    
    @staticmethod
    def create_attitude_controller(config: Dict[str, Any], **kwargs):
        """创建姿态控制器"""
        return AttitudeController(config, **kwargs)
    
    @staticmethod
    def create_angular_rate_controller(config: Dict[str, Any], inertia: Optional[np.ndarray] = None, **kwargs):
        """创建角速度控制器"""
        return AngularRateController(config, inertia=inertia, **kwargs)
    
    @staticmethod
    def create_from_controller_config(controller_config: ControllerConfig, 
                                     drone_mass: float,
                                     drone_inertia: np.ndarray):
        """
        从控制器配置创建所有控制器
        
        Args:
            controller_config: 控制器配置
            drone_mass: 无人机质量
            drone_inertia: 无人机转动惯量
            
        Returns:
            控制器字典
        """
        controllers = {}
        
        # 位置控制器
        if 'position' in controller_config:
            controllers['position'] = PIDControllerFactory.create_position_controller(
                controller_config.position,
                name="PositionController"
            )
        
        # 速度控制器
        if 'velocity' in controller_config:
            controllers['velocity'] = PIDControllerFactory.create_velocity_controller(
                controller_config.velocity,
                mass=drone_mass,
                name="VelocityController"
            )
        
        # 姿态控制器
        if 'attitude' in controller_config:
            controllers['attitude'] = PIDControllerFactory.create_attitude_controller(
                controller_config.attitude,
                name="AttitudeController"
            )
        
        # 角速度控制器
        if 'angle_rate' in controller_config:
            controllers['angle_rate'] = PIDControllerFactory.create_angular_rate_controller(
                controller_config.angle_rate,
                inertia=drone_inertia,
                name="AngularRateController"
            )
        
        return controllers