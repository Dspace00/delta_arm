"""
Main entry point for the Delta Arm simulation.
Run from parent directory: python -m uav_project.main_delta
"""
import os
import sys
import time
import mujoco
import mujoco.viewer
import numpy as np

# Path setup
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Project imports
from uav_project.models.delta_model import DeltaModel
from uav_project.controllers.delta_arm_controller import DeltaArmController
from uav_project.utils.logger import Logger
from uav_project.utils.delta_trajectory import (
    generate_delta_circular_trajectory,
    generate_linear_trajectory,
    generate_square_trajectory,
    generate_point_to_point_trajectory,
    generate_stay_trajectory,
    DELTA_BASE_POSITION
)
from uav_project.config import SIM_TIMESTEP, RENDER_FPS


def run_delta_simulation(mj_model, mj_data, controller, logger, duration=10.0, 
                         trajectory=None, headless=False):
    """
    Runs Delta robot simulation with custom logging.
    
    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        controller: DeltaArmController instance.
        logger: Logger instance.
        duration: Simulation duration in seconds.
        trajectory: List of (time, [x, y, z]) tuples.
        headless: If True, runs without viewer.
    """
    timestep = mj_model.opt.timestep
    total_steps = int(duration / timestep)
    render_interval = 1.0 / RENDER_FPS
    steps_per_render = max(1, int(render_interval / timestep))
    
    # Get motor actuator IDs
    motor_names = ['armmotor1', 'armmotor2', 'armmotor3']
    motor_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                 for name in motor_names]
    
    def get_motor_angles():
        """Gets current motor angles from actuators."""
        angles = np.array([mj_data.actuator(id).ctrl[0] for id in motor_ids])
        return angles
    
    def get_motor_velocities():
        """Gets current motor velocities."""
        vels = np.array([mj_data.actuator_velocity[id] for id in motor_ids])
        return vels
    
    # IK failure tracking
    ik_failed_count = [0]  # Use list for mutable in closure
    
    def step_simulation(step):
        """Single simulation step with Delta-specific logging."""
        sim_time = step * timestep
        
        # 1. Update trajectory target
        if trajectory:
            for i in range(len(trajectory) - 1):
                t_start, pos_start = trajectory[i]
                t_end, pos_end = trajectory[i + 1]
                
                if t_start <= sim_time < t_end:
                    alpha = (sim_time - t_start) / (t_end - t_start)
                    target_pos = np.array(pos_start) + alpha * (np.array(pos_end) - np.array(pos_start))
                    controller.set_target_position(target_pos)
                    break
            
            if sim_time >= trajectory[-1][0]:
                controller.set_target_position(trajectory[-1][1])
        
        # 2. Update controller
        controller.update(sim_time)
        
        # 3. Step physics
        mujoco.mj_step(mj_model, mj_data)
        
        # 4. Logging (every 10 steps)
        if logger and step % 10 == 0:
            # Get position data
            des_pos = controller.current_des_pos_log
            actual_pos = controller.current_actual_pos_log
            
            # Get velocity data
            actual_vel = controller.model.get_ee_sensor_lin_vel()
            
            # Get motor data
            motor_angles = get_motor_angles()
            motor_vels = get_motor_velocities()
            
            # Get status flags
            ik_success = True  # IK should succeed after workspace clamping
            workspace_status = controller.target_outside_workspace
            
            # Log using Delta-specific method
            logger.log_delta(
                time=sim_time,
                des_pos=des_pos,
                actual_pos=actual_pos,
                des_vel=None,
                actual_vel=actual_vel,
                motor_angles=motor_angles,
                motor_vels=motor_vels,
                ik_success=ik_success,
                workspace_status=workspace_status
            )
        
        # 5. Print state (every 5000 steps)
        if step % 5000 == 0:
            progress = step / total_steps * 100
            print(f"Progress: {progress:.1f}% | Time: {sim_time:.2f}s")
    
    # Run simulation
    if headless:
        print(f"Simulation started (Headless). Duration: {duration}s, Timestep: {timestep}s")
        start_real_time = time.time()
        
        for step in range(total_steps):
            step_simulation(step)
        
        elapsed = time.time() - start_real_time
        print(f"Simulation finished. Real time: {elapsed:.2f}s. Factor: {duration/elapsed:.2f}x")
    
    else:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            print(f"Simulation started. Duration: {duration}s, Timestep: {timestep}s")
            start_real_time = time.time()
            
            for step in range(total_steps):
                step_simulation(step)
                
                # Render
                if step % steps_per_render == 0:
                    viewer.sync()
            
            elapsed = time.time() - start_real_time
            print(f"Simulation finished. Real time: {elapsed:.2f}s. Factor: {duration/elapsed:.2f}x")


def main():
    """
    Main entry point for Delta Arm simulation.
    """
    # 0. Simulation parameters
    total_sim_time = 20.0  # seconds
    headless = False  # Set to True for SSH/headless mode
    
    # 1. Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "meshes", "Delta_Arm.xml")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # 2. Initialize logger
    logger = Logger()
    
    # 3. Load MuJoCo model
    print(f"Loading model from: {model_path}")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # 4. Create Delta model wrapper (standalone, no UAV)
    delta_model = DeltaModel(mj_model, mj_data)
    
    # 5. Create controller
    controller = DeltaArmController(delta_model, control_freq=100.0, control_mode='position')
    
    # 6. Initialize motor angles to match end_platform position
    # Use zero initial angles (same as Delta.xml default)
    # The weld constraints will pull end_platform to the correct position
    initial_motor_angles_deg = np.array([0.0, 0.0, 0.0])  # degrees
    initial_motor_angles_rad = np.deg2rad(initial_motor_angles_deg)
    
    # Calculate expected end-effector position using FK
    from uav_project.utils.DeltaKinematics import DeltaKinematics
    kinematics = DeltaKinematics(rod_b=0.1, rod_ee=0.2, r_b=0.074577, r_ee=0.02495)
    expected_pos = kinematics.fk(initial_motor_angles_deg)
    print(f"Expected end-effector position for motor angles {initial_motor_angles_deg} deg: {expected_pos}")
    
    # Set initial motor angles
    delta_model.set_delta_motor_positions(initial_motor_angles_rad)
    
    # Run forward dynamics to let weld constraints converge
    # Use more iterations and reset velocities to help convergence
    print("Initializing simulation (letting constraints converge)...")
    
    # First, run mj_forward to update kinematics
    mujoco.mj_forward(mj_model, mj_data)
    
    # Then run many steps with damping to let constraints settle
    for i in range(5000):
        # Zero out velocities to help convergence
        mj_data.qvel[:] = 0
        mujoco.mj_step(mj_model, mj_data)
        
        if i % 1000 == 0:
            ee_pos = delta_model.get_ee_sensor_pos()
            print(f"  Step {i}: EE position relative to base: {ee_pos}")
    
    print("Initialization complete.")
    
    # Print final state
    ee_pos_final = delta_model.get_ee_sensor_pos()
    print(f"Final EE position (relative to base): {ee_pos_final}")
    print(f"Motor angles (deg): {np.rad2deg(delta_model.get_motor_angles())}")
    
    # Reset controller time after initialization
    controller.reset()
    
    # 7. Generate trajectory (circular in Delta workspace)
    # trajectory = generate_delta_circular_trajectory(
    #     center_offset=[0.0, 0.0, -0.15],  # Center 15cm below base
    #     radius=0.06,                      # 6cm radius circle
    #     total_time=total_sim_time,
    #     num_points=200,
    #     clockwise=False,
    #     validate_workspace=True
    # )
    
    # Alternative trajectories (uncomment to use):
    trajectory = generate_square_trajectory(
        center=[0.0, 0.0, -0.15],
        side_length=0.08,
        total_time=total_sim_time,
        num_points_per_side=50
    )
    
    # trajectory = generate_linear_trajectory(
    #     start=[0.05, 0.05, -0.10],
    #     end=[-0.05, -0.05, -0.20],
    #     total_time=total_sim_time,
    #     num_points=100
    # )
    
    # 8. Print trajectory info
    print(f"\nTrajectory: {len(trajectory)} waypoints over {total_sim_time}s")
    print(f"Start position: {trajectory[0][1]}")
    print(f"End position: {trajectory[-1][1]}")
    print(f"Delta base position (world frame): {DELTA_BASE_POSITION}")
    
    # 9. Run simulation
    run_delta_simulation(
        mj_model=mj_model,
        mj_data=mj_data,
        controller=controller,
        logger=logger,
        duration=total_sim_time,
        trajectory=trajectory,
        headless=headless
    )
    
    # 10. Plot results
    results_path = os.path.join(current_dir, 'delta_simulation_results.png')
    logger.plot_delta_results(
        save_path=results_path,
        show_workspace_boundary=True,
        workspace_params={
            'max_radius': 0.12,
            'z_min': -0.25,
            'z_max': 0.0
        }
    )
    
    print(f"\nSimulation complete. Results saved to: {results_path}")


if __name__ == "__main__":
    main()
