"""
模块功能说明：
UAV控制仿真主程序。
负责初始化仿真环境、控制器，并执行主循环。

使用示例：
python main.py
"""

import os
import time
import yaml
import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import platform

from simulation.mujoco_interface import MujocoSimulator
from simulation.sensor_sim import SimSensor
from controllers.position import PositionController
from controllers.velocity import VelocityController
from controllers.attitude import AttitudeController
from controllers.rate import RateController
from visualization.plotter import Plotter
from trajectory import generate_circular_trajectory, generate_spiral_trajectory, TrajectoryPlanner

def load_config(path="config/pid_params.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config/pid_params.yaml")
    config = load_config(config_path)
    
    # 2. Initialize Simulation
    # Assume meshes/Delta.xml is relative to project root or we find it
    project_root = os.path.dirname(current_dir) # uav_control/..
    # Or strict path:
    model_path = os.path.join(project_root, "code/meshes/Delta.xml")
    if not os.path.exists(model_path):
        # Fallback to local meshes if user moved it
        model_path = os.path.join(current_dir, "../code/meshes/Delta.xml")
    
    print(f"Loading model from: {model_path}")
    sim = MujocoSimulator(model_path)
    sensor = SimSensor(sim.model, sim.data)
    
    # 3. Initialize Controllers
    pos_ctrl = PositionController(config['position'])
    vel_ctrl = VelocityController(config['velocity'], 
                                  mass=config['simulation']['mass'], 
                                  gravity=config['simulation']['gravity'])
    att_ctrl = AttitudeController(config['attitude'])
    rate_ctrl = RateController(config['rate'])
    
    plotter = Plotter()
    
    # 4. Simulation Parameters
    dt = config['simulation']['dt']
    duration = 20.0 # seconds
    total_steps = int(duration / dt)

    # Initialize Trajectory Planner
    # Option 1: Circular Trajectory
    center = [0.0, 0.0, 1.5]
    radius = 1.0
    
    # 0-10s: Circular
    traj_points = generate_circular_trajectory(
        center=center, 
        radius=radius, 
        total_time=10.0, 
        num_points=100,
        start_angle=0,
        clockwise=False,
        height_variation=True,
        height_amplitude=0.2
    )
    
    # Option 2: Spiral Trajectory (Uncomment to use)
    # traj_points = generate_spiral_trajectory(
    #     center=[0, 0, 0.5],
    #     start_radius=0.5,
    #     end_radius=2.0,
    #     total_time=15.0,
    #     num_turns=3,
    #     num_points=150
    # )
    
    planner = TrajectoryPlanner(traj_points)
    
    # Frequencies
    freq_pos = 50
    freq_vel = 50
    freq_att = 250
    freq_rate = 1000
    
    # Timers
    last_pos_time = 0.0
    last_vel_time = 0.0
    last_att_time = 0.0
    last_rate_time = 0.0
    
    # State Variables (Buffers)
    target_pos = np.array([0.0, 0.0, 1.0])
    target_vel = np.array([0.0, 0.0, 0.0])
    target_att_quat = np.quaternion(1, 0, 0, 0) # w,x,y,z
    target_thrust = np.zeros(3)
    target_rate = np.array([0.0, 0.0, 0.0])
    cmd_wrench = np.zeros(6) # Fx, Fy, Fz, Mx, My, Mz 
    
    sim_time = 0.0
    
    print("Starting simulation...")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        start_wall_time = time.time()
        
        for step in range(total_steps):
            sim_time = step * dt
            
            # 1. Sensor Data Acquisition
            current_pos, current_vel, current_att_quat, current_rate = sensor.get_state()
            
            # 2. Controller Update (Cascade)
            
            # Position Control (50Hz)
            if sim_time - last_pos_time >= 1.0/freq_pos:
                # Update Target Pos (Trajectory Generation)
                target_pos = planner.get_target_position(sim_time)
                    
                target_vel = pos_ctrl.update(target_pos, current_pos)
                last_pos_time = sim_time
            
            # Velocity Control (50Hz)
            if sim_time - last_vel_time >= 1.0/freq_vel:
                target_att_quat, target_thrust = vel_ctrl.update(target_vel, current_vel)
                last_vel_time = sim_time
                
            # Attitude Control (250Hz)
            if sim_time - last_att_time >= 1.0/freq_att:
                target_rate = att_ctrl.update(target_att_quat, current_att_quat)
                last_att_time = sim_time
                
            # Rate Control (1000Hz)
            if sim_time - last_rate_time >= 1.0/freq_rate:
                cmd_torques = rate_ctrl.update(target_rate, current_rate)
                # Combine Thrust and Torques
                # Note: Velocity controller calculates Total Thrust Magnitude (scalar) aligned with body Z (ideally)
                # We need to apply this force in body frame.
                # Assuming body z-axis is the thrust direction.
                cmd_force = target_thrust
                
                sim.apply_wrench(cmd_force, cmd_torques)
                last_rate_time = sim_time
                
                # Record Data (at rate freq)
                # Convert Quats to Euler for plotting
                att_euler = R.from_quat([current_att_quat.x, current_att_quat.y, current_att_quat.z, current_att_quat.w]).as_euler('xyz', degrees=True)
                target_att_euler = R.from_quat([target_att_quat.x, target_att_quat.y, target_att_quat.z, target_att_quat.w]).as_euler('xyz', degrees=True)
                
                plotter.update(sim_time, current_pos, target_pos, current_vel, target_vel, 
                               att_euler, target_att_euler, current_rate, target_rate, 
                               target_thrust, cmd_torques)
            
            # 4. Simulation Step
            sim.step()
            
            # Sync Viewer (60Hz)
            if step % int(1.0/dt / 60.0) == 0:
                viewer.sync()
                
        # End Loop
        print("Simulation finished.")
        plotter.plot("uav_control_results.png")

if __name__ == "__main__":
    main()
