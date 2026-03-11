#!/usr/bin/env python3
"""
集成仿真系统主程序
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from drone_simulator.simulation.integrated_simulator import IntegratedDroneSimulator
from drone_simulator.trajectory import generate_circular_trajectory, generate_spiral_trajectory


def test_position_control():
    """测试位置控制"""
    print("=== Testing Position Control ===")
    
    # 创建仿真器
    simulator = IntegratedDroneSimulator("meshes/Delta.xml", config_path="./configs")
    
    # 设置初始状态
    simulator.set_initial_state(
        position=[0, 0, 1.0],
        attitude=[1, 0, 0, 0]
    )
    
    # 设置控制模式为位置控制
    simulator.controller.set_control_mode('position')
    
    # 设置目标位置
    simulator.controller.set_target(position=[2.0, 1.0, 1.5])
    
    # 运行仿真
    simulator.run(duration=5.0, real_time=False)
    
    # 绘制结果
    simulator.plot_results()
    
    return simulator


def test_circular_trajectory():
    """测试圆形轨迹"""
    print("\n=== Testing Circular Trajectory ===")
    
    # 创建仿真器
    simulator = IntegratedDroneSimulator("meshes/Delta.xml", config_path="./configs/default")
    
    # 设置初始状态
    simulator.set_initial_state(
        position=[0, 0, 1.0],
        attitude=[1, 0, 0, 0]
    )
    
    # 设置控制模式
    simulator.controller.set_control_mode('position')
    
    # 生成圆形轨迹
    trajectory = generate_circular_trajectory(
        center=[0, 0, 1.5],
        radius=2.0,
        total_time=10.0,
        num_points=100,
        clockwise=False,
        height_variation=True,
        height_amplitude=0.5
    )
    
    # 运行仿真
    simulator.run(duration=12.0, real_time=False, trajectory=trajectory)
    
    # 绘制结果
    simulator.plot_results()
    
    return simulator


def test_spiral_trajectory():
    """测试螺旋轨迹"""
    print("\n=== Testing Spiral Trajectory ===")
    
    # 创建仿真器
    simulator = IntegratedDroneSimulator("meshes/Delta.xml", config_path="./configs/default")
    
    # 设置初始状态
    simulator.set_initial_state(
        position=[0, 0, 1.0],
        attitude=[1, 0, 0, 0]
    )
    
    # 设置控制模式
    simulator.controller.set_control_mode('position')
    
    # 生成螺旋轨迹
    trajectory = generate_spiral_trajectory(
        center=[0, 0, 1.0],
        start_radius=0.5,
        end_radius=3.0,
        total_time=15.0,
        num_turns=3,
        num_points=150,
        clockwise=False
    )
    
    # 运行仿真
    simulator.run(duration=16.0, real_time=False, trajectory=trajectory)
    
    # 绘制结果
    simulator.plot_results()
    
    return simulator


def test_waypoint_trajectory():
    """测试航点轨迹"""
    print("\n=== Testing Waypoint Trajectory ===")
    
    # 创建仿真器
    simulator = IntegratedDroneSimulator("meshes/Delta.xml", config_path="./configs/default")
    
    # 设置初始状态
    simulator.set_initial_state(
        position=[0, 0, 1.0],
        attitude=[1, 0, 0, 0]
    )
    
    # 设置控制模式
    simulator.controller.set_control_mode('position')
    
    # 定义航点
    waypoints = [
        (0.0, [0, 0, 1.0]),
        (2.0, [2, 0, 1.5]),
        (5.0, [2, 2, 2.0]),
        (8.0, [0, 2, 1.5]),
        (11.0, [0, 0, 1.0]),
        (13.0, [-2, 0, 1.5]),
        (16.0, [-2, -2, 2.0]),
        (19.0, [0, -2, 1.5]),
        (22.0, [0, 0, 1.0])
    ]
    
    # 规划轨迹
    simulator.plan_trajectory('waypoints', waypoints=waypoints, smooth=True)
    
    # 运行仿真
    simulator.run(duration=24.0, real_time=False)
    
    # 绘制结果
    simulator.plot_results()
    
    return simulator


def test_different_control_modes():
    """测试不同的控制模式"""
    print("\n=== Testing Different Control Modes ===")
    
    # 创建仿真器
    simulator = IntegratedDroneSimulator("meshes/Delta.xml", config_path="./configs/default")
    
    # 设置初始状态
    simulator.set_initial_state(
        position=[0, 0, 1.0],
        attitude=[1, 0, 0, 0]
    )
    
    print("\n1. Testing Rate Control Mode")
    simulator.controller.set_control_mode('rate')
    simulator.controller.set_target(angular_rate=[0.5, 0.0, 0.0])
    simulator.run(duration=2.0, real_time=False)
    
    print("\n2. Testing Attitude Control Mode")
    simulator.set_initial_state(position=[0, 0, 1.0])
    simulator.controller.set_control_mode('attitude')
    simulator.controller.set_target(attitude=[0.707, 0, 0.707, 0])  # 45度俯仰
    simulator.run(duration=3.0, real_time=False)
    
    print("\n3. Testing Velocity Control Mode")
    simulator.set_initial_state(position=[0, 0, 1.0])
    simulator.controller.set_control_mode('velocity')
    simulator.controller.set_target(velocity=[1.0, 0.0, 0.0])
    simulator.run(duration=4.0, real_time=False)
    
    print("\n4. Testing Position Control Mode")
    simulator.set_initial_state(position=[0, 0, 1.0])
    simulator.controller.set_control_mode('position')
    simulator.controller.set_target(position=[3.0, 2.0, 1.5])
    simulator.run(duration=6.0, real_time=False)
    
    # 绘制结果
    simulator.plot_results()
    
    return simulator


def main():
    """主函数"""
    print("=== Integrated Drone Simulation System ===")
    
    # 测试选项
    tests = {
        '1': ('Position Control', test_position_control),
        '2': ('Circular Trajectory', test_circular_trajectory),
        '3': ('Spiral Trajectory', test_spiral_trajectory),
        '4': ('Waypoint Trajectory', test_waypoint_trajectory),
        '5': ('All Control Modes', test_different_control_modes)
    }
    
    # 显示菜单
    print("\nAvailable Tests:")
    for key, (name, _) in tests.items():
        print(f"  {key}. {name}")
    print("  0. Exit")
    
    # 获取用户选择
    while True:
        choice = input("\nSelect test (0-5): ").strip()
        
        if choice == '0':
            print("Exiting...")
            break
        
        if choice in tests:
            name, test_func = tests[choice]
            print(f"\nRunning {name}...")
            try:
                test_func()
            except Exception as e:
                print(f"Error during test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Invalid choice. Please select 0-5.")


if __name__ == "__main__":
    main()