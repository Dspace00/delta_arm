#!/usr/bin/env python3
"""
测试新的PID控制器系统
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import quaternion
from drone_simulator.controllers import *
from drone_simulator.utils import ConfigManager


def test_pid_controller():
    """测试通用PID控制器"""
    print("=== Testing PID Controller ===")
    
    # 创建PID参数
    params = PIDParams(
        kp=[2.0, 2.0, 2.0],
        ki=[0.1, 0.1, 0.1],
        kd=[0.01, 0.01, 0.01],
        dt=0.02,
        output_limit=(-10.0, 10.0)
    )
    
    # 创建PID控制器
    pid = PIDController(params, name="TestPID")
    print(f"Created: {pid}")
    
    # 模拟控制
    setpoint = np.array([1.0, 2.0, 3.0])
    measurement = np.array([0.0, 0.0, 0.0])
    
    for i in range(10):
        output = pid.update(setpoint, measurement)
        measurement += output * 0.1  # 简单积分
        print(f"Step {i}: Output = {output}, Measurement = {measurement}")
    
    # 重置
    pid.reset()
    print("PID reset successfully")
    
    return pid


def test_position_controller():
    """测试位置控制器"""
    print("\n=== Testing Position Controller ===")
    
    config = {
        'frequency': 50,
        'pid': {
            'kp': [1.0, 1.0, 1.5],
            'ki': [0.0, 0.0, 0.0],
            'kd': [0.0, 0.0, 0.0],
            'output_limit': [-2.0, 2.0]
        }
    }
    
    pos_ctrl = PositionController(config, name="TestPosition")
    print(f"Created: {pos_ctrl}")
    
    # 模拟控制
    target_pos = np.array([5.0, 3.0, 10.0])
    current_pos = np.array([0.0, 0.0, 1.0])
    
    velocity_cmd = pos_ctrl.update(target_pos, current_pos)
    print(f"Position error: {target_pos - current_pos}")
    print(f"Velocity command: {velocity_cmd}")
    
    return pos_ctrl


def test_velocity_controller():
    """测试速度控制器"""
    print("\n=== Testing Velocity Controller ===")
    
    config = {
        'frequency': 50,
        'pid': {
            'kp': [0.5, 0.5, 1.0],
            'ki': [0.01, 0.01, 0.02],
            'kd': [0.05, 0.05, 0.1],
            'output_limit': [-5.0, 5.0]
        }
    }
    
    vel_ctrl = VelocityController(config, mass=1.27, name="TestVelocity")
    print(f"Created: {vel_ctrl}")
    
    # 模拟控制
    target_vel = np.array([1.0, 0.5, 0.0])
    current_vel = np.array([0.0, 0.0, 0.0])
    
    accel_cmd = vel_ctrl.update(target_vel, current_vel)
    thrust_vector = vel_ctrl.calculate_thrust_vector(accel_cmd)
    
    print(f"Acceleration command: {accel_cmd}")
    print(f"Thrust vector: {thrust_vector}")
    print(f"Thrust magnitude: {np.linalg.norm(thrust_vector):.2f} N")
    
    return vel_ctrl


def test_attitude_controller():
    """测试姿态控制器"""
    print("\n=== Testing Attitude Controller ===")
    
    config = {
        'frequency': 250,
        'pid': {
            'kp': [5.0, 5.0, 3.0],
            'ki': [0.0, 0.0, 0.0],
            'kd': [0.0, 0.0, 0.0],
            'output_limit': [-2.0, 2.0]
        }
    }
    
    att_ctrl = AttitudeController(config, name="TestAttitude")
    print(f"Created: {att_ctrl}")
    
    # 测试姿态计算
    thrust_vector = np.array([0.0, 0.0, 15.0])  # 垂直向上
    target_yaw = np.deg2rad(45.0)  # 45度偏航
    
    desired_att = att_ctrl.calculate_desired_attitude(thrust_vector, target_yaw)
    print(f"Desired attitude (quaternion): {desired_att}")
    
    # 测试姿态控制
    current_att = quaternion.from_float_array([1, 0, 0, 0])  # 无旋转
    rate_cmd = att_ctrl.update(desired_att, current_att)
    print(f"Angular rate command: {rate_cmd}")
    
    return att_ctrl


def test_angular_rate_controller():
    """测试角速度控制器"""
    print("\n=== Testing Angular Rate Controller ===")
    
    config = {
        'frequency': 1000,
        'pid': {
            'kp': [0.1, 0.1, 0.2],
            'ki': [0.01, 0.01, 0.02],
            'kd': [0.001, 0.001, 0.002],
            'output_limit': [-1.0, 1.0]
        }
    }
    
    inertia = np.array([0.022, 0.022, 0.036])
    rate_ctrl = AngularRateController(config, inertia=inertia, name="TestAngularRate")
    print(f"Created: {rate_ctrl}")
    
    # 模拟控制
    target_rate = np.array([0.5, 0.0, 0.0])  # 滚转0.5 rad/s
    current_rate = np.array([0.0, 0.0, 0.0])
    
    torque_cmd = rate_ctrl.update(target_rate, current_rate)
    print(f"Torque command: {torque_cmd}")
    
    return rate_ctrl


def test_mixer():
    """测试混控器"""
    print("\n=== Testing Mixer ===")
    
    from drone_simulator.models.mixer import Mixer
    
    mixer_config = {
        'thrust_coefficient': 0.01343,
        'torque_coefficient': 3.099e-4,
        'arm_length': 0.18,
        'max_thrust': 6.5,
        'max_torque': 0.15,
        'max_speed': 22
    }
    
    mixer = Mixer(mixer_config)
    print(f"Created: {mixer}")
    
    # 测试动力分配
    thrust = 10.0  # N
    torque = np.array([0.1, 0.05, 0.02])  # Nm
    
    motor_speeds = mixer.allocate(thrust, torque)
    print(f"Motor speeds (krpm): {motor_speeds}")
    
    # 验证
    thrust_calc = np.sum(mixer.krpm_to_thrust(motor_speeds))
    print(f"Calculated thrust: {thrust_calc:.2f} N (target: {thrust} N)")
    
    return mixer


def test_controller_factory():
    """测试控制器工厂"""
    print("\n=== Testing Controller Factory ===")
    
    # 创建配置管理器
    config_manager = ConfigManager()
    config_manager.load_default()
    
    # 使用工厂创建控制器
    controllers = PIDControllerFactory.create_from_controller_config(
        controller_config=config_manager.controller,
        drone_mass=config_manager.drone.mass,
        drone_inertia=np.array([config_manager.drone.inertia['Ixx'],
                               config_manager.drone.inertia['Iyy'],
                               config_manager.drone.inertia['Izz']])
    )
    
    print(f"Created {len(controllers)} controllers:")
    for name, controller in controllers.items():
        print(f"  - {name}: {controller}")
    
    return controllers


def run_integration_test():
    """运行集成测试"""
    print("\n=== Running Integration Test ===")
    
    # 创建所有控制器
    config_manager = ConfigManager()
    config_manager.load_default()
    
    controllers = PIDControllerFactory.create_from_controller_config(
        config_manager.controller,
        config_manager.drone.mass,
        np.array([config_manager.drone.inertia['Ixx'],
                 config_manager.drone.inertia['Iyy'],
                 config_manager.drone.inertia['Izz']])
    )
    
    # 创建混控器
    from drone_simulator.models.mixer import Mixer
    mixer = Mixer(config_manager.drone.motors)
    
    # 模拟控制循环
    print("\nSimulating control loop...")
    
    # 初始状态
    position = np.array([0.0, 0.0, 1.0])
    velocity = np.array([0.0, 0.0, 0.0])
    attitude = quaternion.from_float_array([1, 0, 0, 0])
    angular_rate = np.array([0.0, 0.0, 0.0])
    
    # 目标
    target_position = np.array([5.0, 3.0, 10.0])
    
    dt = 0.02  # 50 Hz
    steps = 5
    
    for step in range(steps):
        print(f"\nStep {step}:")
        print(f"  Position: {position}")
        print(f"  Target: {target_position}")
        
        # 位置控制
        if 'position' in controllers:
            velocity_target = controllers['position'].update(target_position, position)
            print(f"  Velocity target: {velocity_target}")
        
        # 速度控制
        if 'velocity' in controllers:
            acceleration = controllers['velocity'].update(velocity_target, velocity, dt)
            thrust_vector = controllers['velocity'].calculate_thrust_vector(acceleration)
            print(f"  Thrust vector: {thrust_vector}")
            
            # 姿态计算
            if 'attitude' in controllers:
                target_attitude = controllers['attitude'].calculate_desired_attitude(
                    thrust_vector, target_yaw=0.0
                )
                print(f"  Target attitude: {target_attitude}")
        
        # 简单更新（模拟物理）
        position += velocity * dt
        velocity += acceleration * dt
        
        print(f"  New position: {position}")
    
    print("\nIntegration test completed!")


if __name__ == "__main__":
    print("Testing New PID Controller System\n")
    
    # 运行测试
    test_pid_controller()
    test_position_controller()
    test_velocity_controller()
    test_attitude_controller()
    test_angular_rate_controller()
    test_mixer()
    test_controller_factory()
    
    # 集成测试
    run_integration_test()
    
    print("\n=== All PID Controller Tests Completed ===")