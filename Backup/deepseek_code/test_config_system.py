#!/usr/bin/env python3
"""
测试配置系统
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_simulator.utils import ConfigManager, get_global_config


def test_config_creation():
    """测试配置创建"""
    print("=== Testing Config Creation ===")
    
    # 创建配置管理器
    config_manager = ConfigManager("./test_configs")
    
    # 加载默认配置
    config_manager.load_default()
    
    # 验证配置
    is_valid = config_manager.validate_all()
    print(f"Config validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # 保存配置
    config_manager.save_configs("test")
    print("Configs saved to ./test_configs")
    
    # 打印配置
    print("\n=== Drone Config ===")
    print(config_manager.drone.to_yaml())
    
    print("\n=== Controller Config ===")
    print(config_manager.controller.to_yaml()[:500] + "...")
    
    return config_manager


def test_config_loading():
    """测试配置加载"""
    print("\n=== Testing Config Loading ===")
    
    # 加载刚才保存的配置
    config_manager = ConfigManager("./test_configs")
    config_manager.load_from_files(
        "./test_configs/test_drone.yaml",
        "./test_configs/test_controller.yaml",
        "./test_configs/test_simulation.yaml"
    )
    
    print("Config loaded successfully")
    print(f"Drone mass: {config_manager.drone.mass} kg")
    print(f"Position controller frequency: {config_manager.controller.position['frequency']} Hz")
    
    return config_manager


def test_global_config():
    """测试全局配置"""
    print("\n=== Testing Global Config ===")
    
    # 获取全局配置
    global_config = get_global_config()
    
    print("Global config loaded")
    print(f"Simulation duration: {global_config.simulation.duration} s")
    
    # 修改配置
    global_config.simulation.duration = 20.0
    print(f"Modified duration: {global_config.simulation.duration} s")
    
    return global_config


def test_config_updates():
    """测试配置更新"""
    print("\n=== Testing Config Updates ===")
    
    config_manager = ConfigManager()
    config_manager.load_default()
    
    # 更新部分配置
    updates = {
        'drone': {
            'mass': 2.0,
            'dimensions': {
                'arm_length': 0.25
            }
        },
        'controller': {
            'position': {
                'frequency': 100
            }
        }
    }
    
    # 使用字典更新
    combined = config_manager.get_combined_config()
    
    # 深度更新
    def deep_update(original, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                deep_update(original[key], value)
            else:
                original[key] = value
    
    deep_update(combined, updates)
    config_manager.load_from_dict(combined)
    
    print(f"Updated drone mass: {config_manager.drone.mass} kg")
    print(f"Updated arm length: {config_manager.drone.dimensions['arm_length']} m")
    print(f"Updated position frequency: {config_manager.controller.position['frequency']} Hz")


def test_invalid_config():
    """测试无效配置"""
    print("\n=== Testing Invalid Config ===")
    
    config_manager = ConfigManager()
    
    # 故意创建无效配置
    invalid_config = {
        'drone': {
            'mass': -1.0,  # 无效质量
            'dimensions': {
                'arm_length': -0.1  # 无效臂长
            }
        }
    }
    
    try:
        config_manager.load_from_dict(invalid_config)
        config_manager.validate_all()
    except ValueError as e:
        print(f"Expected validation error: {e}")
        print("Invalid config test PASSED")


if __name__ == "__main__":
    print("Testing Configuration System\n")
    
    # 运行测试
    test_config_creation()
    test_config_loading()
    test_global_config()
    test_config_updates()
    test_invalid_config()
    
    print("\n=== All Configuration Tests Completed ===")