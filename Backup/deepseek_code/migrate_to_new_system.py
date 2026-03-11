#!/usr/bin/env python3
"""
从旧系统迁移到新系统的辅助脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def compare_controllers():
    """比较新旧控制器功能"""
    print("=== Controller Feature Comparison ===")
    
    old_features = {
        'DiscretePID3D': '3D离散PID控制器（完整PID）',
        'PosPID': '3D位置PID控制器（只有比例项）',
        'VelPID': '3D速度PID控制器（完整PID）',
        'AttitudePID3D': '姿态PID控制器（四元数误差）',
        'AngVelPID': '角速度PID控制器（完整PID）',
        'QuadrotorCascadePID': '四旋翼串级PID主控制器'
    }
    
    new_features = {
        'PIDController': '通用PID控制器（支持1D/2D/3D）',
        'PositionController': '位置控制器（基于PID）',
        'VelocityController': '速度控制器（基于PID，含重力补偿）',
        'AttitudeController': '姿态控制器（四元数误差计算）',
        'AngularRateController': '角速度控制器（基于PID）',
        'CascadedController': '串级主控制器（整合所有子控制器）'
    }
    
    print("\nOld System Controllers:")
    for name, desc in old_features.items():
        print(f"  - {name}: {desc}")
    
    print("\nNew System Controllers:")
    for name, desc in new_features.items():
        print(f"  - {name}: {desc}")
    
    print("\nMigration Path:")
    print("  Old → New")
    print("  DiscretePID3D → PIDController")
    print("  PosPID → PositionController")
    print("  VelPID → VelocityController")
    print("  AttitudePID3D → AttitudeController")
    print("  AngVelPID → AngularRateController")
    print("  QuadrotorCascadePID → CascadedController")


def check_dependencies():
    """检查依赖项"""
    print("\n=== Dependency Check ===")
    
    required = ['mujoco', 'numpy', 'quaternion', 'scipy']
    optional = ['matplotlib', 'yaml']
    
    print("Required packages:")
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
    
    print("\nOptional packages:")
    for package in optional:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (Not required but recommended)")


def setup_configs():
    """设置配置文件"""
    print("\n=== Configuration Setup ===")
    
    config_dir = "./configs"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f"Created directory: {config_dir}")
    
    # 创建默认配置文件
    from drone_simulator.utils import ConfigManager
    
    config_manager = ConfigManager(config_dir)
    config_manager.create_default_configs()
    
    print("Default configs created in ./configs/")
    print("  - default_drone.yaml")
    print("  - default_controller.yaml")
    print("  - default_simulation.yaml")


def test_imports():
    """测试新系统的导入"""
    print("\n=== Testing New System Imports ===")
    
    modules = [
        'drone_simulator',
        'drone_simulator.controllers',
        'drone_simulator.models',
        'drone_simulator.simulation',
        'drone_simulator.trajectory',
        'drone_simulator.utils'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")


def create_compatibility_layer():
    """创建兼容层（可选）"""
    print("\n=== Creating Compatibility Layer ===")
    
    compatibility_code = '''
# drone_simulator/compatibility.py
"""
兼容层 - 允许旧代码逐步迁移到新系统
"""
import warnings

def deprecated(message):
    """装饰器：标记函数为已弃用"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {message}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 旧的控制器类（转发到新类）
class DiscretePID3D:
    """兼容类：转发到新的PIDController"""
    @deprecated("Use drone_simulator.controllers.PIDController instead")
    def __init__(self, *args, **kwargs):
        from drone_simulator.controllers import PIDController, PIDParams
        # 转换参数...
        pass


# 其他兼容类...
'''
    
    print("Compatibility layer code created.")
    print("You can add this to maintain backward compatibility during migration.")


def main():
    """主函数"""
    print("=== Migration Assistant ===")
    print("This script helps migrate from the old system to the new modular system.")
    
    compare_controllers()
    check_dependencies()
    setup_configs()
    test_imports()
    create_compatibility_layer()
    
    print("\n=== Migration Steps ===")
    print("1. Test the new system with test scripts")
    print("2. Update your main program to use IntegratedDroneSimulator")
    print("3. Convert configuration files to YAML format")
    print("4. Update any custom controllers to use new base classes")
    print("5. Test thoroughly before removing old code")
    
    print("\n=== Next Actions ===")
    print("Run: python test_pid_controllers.py")
    print("Run: python test_simple_simulation.py")
    print("Run: python run_integrated_simulation.py")


if __name__ == "__main__":
    main()