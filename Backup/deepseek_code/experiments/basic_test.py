"""
基础测试脚本
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from drone_simulator.simulation.mujoco_env import MuJoCoEnvironment
from drone_simulator.models.drone import DroneModel, DroneState
import quaternion


def test_mujoco_environment():
    """测试MuJoCo环境"""
    print("Testing MuJoCo Environment...")
    
    # 创建仿真环境
    env = MuJoCoEnvironment("Delta.xml")
    
    try:
        # 启动查看器
        env.launch_viewer()
        
        # 运行仿真几步
        for i in range(100):
            env.step()
            env.sync_viewer()
            time.sleep(0.01)
            
    finally:
        env.close_viewer()
    
    print("Test completed!")


def test_drone_model():
    """测试无人机模型"""
    print("Testing Drone Model...")
    
    # 创建无人机模型
    drone = DroneModel(mass=1.27, arm_length=0.18)
    
    # 创建状态
    state = DroneState(
        position=np.array([0.0, 0.0, 1.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        attitude=quaternion.from_float_array([1, 0, 0, 0]),  # [w, x, y, z]
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        timestamp=0.0
    )
    
    # 更新状态
    drone.update_state(state)
    
    print(f"Drone state: {drone.state}")
    print(f"Hover thrust: {drone.calculate_thrust_to_hover():.2f} N")
    print("Test completed!")


if __name__ == "__main__":
    # 运行测试
    print("=== Basic Tests ===")
    # test_drone_model()
    # test_mujoco_environment()
    print("All tests completed!")