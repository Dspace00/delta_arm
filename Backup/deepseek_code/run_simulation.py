#!/usr/bin/env python3
"""
无人机仿真主入口 - 更新版本
"""
import sys
import os

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置系统
from drone_simulator.utils import get_global_config


def main():
    """主函数"""
    print("=== UAV Simulation with Config System ===")
    
    # 加载配置
    print("\n1. Loading configuration...")
    config = get_global_config()
    
    # 显示配置信息
    print(f"   Simulation: {config.simulation.name}")
    print(f"   Duration: {config.simulation.duration} s")
    print(f"   Drone mass: {config.drone.mass} kg")
    print(f"   Control mode: {config.controller.control_mode}")
    
    # 验证配置
    print("\n2. Validating configuration...")
    if config.validate_all():
        print("   ✓ All configurations are valid")
    else:
        print("   ✗ Some configurations are invalid")
        return
    
    # 运行仿真（这里暂时调用原有代码）
    print("\n3. Starting simulation...")
    
    # 临时兼容：使用原有代码
    try:
        from uav_code import OptimizedDroneSimulation
        simulator = OptimizedDroneSimulation(config.simulation.model_path)
        
        # 将配置传递给仿真器（需要修改OptimizedDroneSimulation以接受配置）
        # 暂时使用硬编码参数
        simulator.run_simulation(duration=config.simulation.duration)
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure the original uav_code.py is available")
    
    print("\n=== Simulation Finished ===")


if __name__ == "__main__":
    main()