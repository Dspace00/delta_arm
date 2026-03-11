"""
配置加载工具
"""
import os
import yaml
import json
from datetime import datetime
from typing import Dict, Any, Union, Optional
from pathlib import Path
from ..config import *


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "./configs"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置目录路径
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前配置
        self.drone: Optional[DroneConfig] = None
        self.controller: Optional[ControllerConfig] = None
        self.simulation: Optional[SimulationConfig] = None
        
    def load_default(self):
        """加载默认配置"""
        self.drone = DroneConfig()
        self.controller = ControllerConfig()
        self.simulation = SimulationConfig()
        return self
    
    def load_from_files(self, drone_config: str = None, 
                       controller_config: str = None, 
                       simulation_config: str = None):
        """
        从文件加载配置
        
        Args:
            drone_config: 无人机配置文件路径
            controller_config: 控制器配置文件路径
            simulation_config: 仿真配置文件路径
        """
        if drone_config:
            self.drone = DroneConfig.load(drone_config)
        
        if controller_config:
            self.controller = ControllerConfig.load(controller_config)
        
        if simulation_config:
            self.simulation = SimulationConfig.load(simulation_config)
        
        # 如果没有提供某个配置，使用默认
        if self.drone is None:
            self.drone = DroneConfig()
        if self.controller is None:
            self.controller = ControllerConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()
        
        return self
    
    def load_from_dict(self, config_dict: Dict[str, Any]):
        """
        从字典加载配置
        
        Args:
            config_dict: 配置字典
        """
        if 'drone' in config_dict:
            self.drone = DroneConfig(config_dict['drone'])
        
        if 'controller' in config_dict:
            self.controller = ControllerConfig(config_dict['controller'])
        
        if 'simulation' in config_dict:
            self.simulation = SimulationConfig(config_dict['simulation'])
        
        # 确保所有配置都已加载
        self.load_default()
        
        return self
    
    def save_configs(self, prefix: str = ""):
        """
        保存当前配置到文件
        
        Args:
            prefix: 文件名前缀
        """
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{prefix}" # if prefix else timestamp
        
        if self.drone:
            self.drone.save(self.config_dir / f"{prefix}_drone.yaml")
        
        if self.controller:
            self.controller.save(self.config_dir / f"{prefix}_controller.yaml")
        
        if self.simulation:
            self.simulation.save(self.config_dir / f"{prefix}_simulation.yaml")
    
    def create_default_configs(self):
        """创建默认配置文件"""
        self.load_default()
        self.save_configs("default")
        print(f"Default configs saved to {self.config_dir}")
    
    def get_combined_config(self) -> Dict[str, Any]:
        """获取合并的配置字典"""
        return {
            'drone': self.drone.to_dict() if self.drone else {},
            'controller': self.controller.to_dict() if self.controller else {},
            'simulation': self.simulation.to_dict() if self.simulation else {}
        }
    
    def validate_all(self) -> bool:
        """验证所有配置"""
        valid = True
        
        if self.drone:
            try:
                self.drone.validate()
                print("✓ Drone config is valid")
            except Exception as e:
                print(f"✗ Drone config error: {e}")
                valid = False
        
        if self.controller:
            try:
                self.controller.validate()
                print("✓ Controller config is valid")
            except Exception as e:
                print(f"✗ Controller config error: {e}")
                valid = False
        
        if self.simulation:
            try:
                self.simulation.validate()
                print("✓ Simulation config is valid")
            except Exception as e:
                print(f"✗ Simulation config error: {e}")
                valid = False
        
        return valid
    
    def __str__(self):
        """字符串表示"""
        return f"ConfigManager:\n" \
               f"  Drone: {'Loaded' if self.drone else 'Not loaded'}\n" \
               f"  Controller: {'Loaded' if self.controller else 'Not loaded'}\n" \
               f"  Simulation: {'Loaded' if self.simulation else 'Not loaded'}"


def load_global_config(config_path: str = None) -> ConfigManager:
    """
    加载全局配置
    
    Args:
        config_path: 配置文件路径（可以是YAML或JSON）
        
    Returns:
        ConfigManager实例
    """
    config_manager = ConfigManager()
    
    if config_path and os.path.exists(config_path):
        # 尝试加载统一配置文件
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {config_path}")
            
            config_manager.load_from_dict(config_dict)
            print(f"Loaded config from {config_path}")
        except Exception as e:
            print(f"Failed to load config from {config_path}: {e}")
            print("Using default configs")
            config_manager.load_default()
    else:
        # 尝试加载单独的配置文件
        drone_config = config_manager.config_dir / "default_drone.yaml"
        controller_config = config_manager.config_dir / "default_controller.yaml"
        simulation_config = config_manager.config_dir / "default_simulation.yaml"
        
        config_manager.load_from_files(
            drone_config if drone_config.exists() else None,
            controller_config if controller_config.exists() else None,
            simulation_config if simulation_config.exists() else None
        )
    
    return config_manager


# 创建全局配置管理器实例（单例模式）
_global_config_manager = None

def get_global_config(config_path: str = None) -> ConfigManager:
    """获取全局配置管理器（单例）"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = load_global_config(config_path)
    return _global_config_manager