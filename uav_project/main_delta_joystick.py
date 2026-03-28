"""
Main entry point for Delta Arm joystick-controlled simulation.

This script enables real-time control of the Delta robot arm using a 
PS2 2.4G wireless gamepad. The joystick inputs are mapped to the 
robot's workspace coordinates for intuitive manual control.

Hardware Requirements:
    - PS2 2.4G wireless gamepad with USB receiver
    - Device name: USB WirelessGamepad

Controls:
    - Left Stick: Control X/Y position of end-effector
    - Right Stick Y: Control Z position of end-effector
    - START: Exit simulation
    - SELECT: Reset position to center

Run from parent directory: python -m uav_project.main_delta_joystick
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
from uav_project.hardware.ps2_controller import PS2Controller
from uav_project.config import SIM_TIMESTEP, RENDER_FPS


def run_joystick_simulation(mj_model, mj_data, controller, logger, 
                            ps2_controller, duration=60.0, headless=False):
    """
    Runs Delta robot simulation with joystick control.
    
    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data.
        controller: DeltaArmController instance.
        logger: Logger instance.
        ps2_controller: PS2Controller instance.
        duration: Maximum simulation duration in seconds.
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
    
    # Simulation state
    running = True
    sim_time = 0.0
    
    print("\n" + "="*50)
    print("Joystick Control Active")
    print("="*50)
    print("Controls:")
    print("  Left Stick:  X/Y position")
    print("  Right Stick: Z position")
    print("  SELECT:      Reset to center")
    print("  START:       Exit simulation")
    print("="*50 + "\n")
    
    def step_simulation(step):
        """Single simulation step with joystick input."""
        nonlocal running
        
        sim_time = step * timestep
        
        # 1. Read joystick input
        ps2_controller.read_input()
        
        # 2. Check exit condition
        if ps2_controller.get_button('start'):
            print("\nSTART button pressed. Exiting...")
            running = False
            return
        
        # 3. Check reset condition
        if ps2_controller.get_button('select'):
            ps2_controller.reset_position()
            # Small delay to prevent repeated resets
            time.sleep(0.1)
        
        # 4. Get target position from joystick
        target_pos = ps2_controller.get_position()
        
        # 5. Update controller with target
        controller.set_target_position(target_pos)
        
        # 6. Update controller (IK calculation)
        controller.update(sim_time)
        
        # 7. Step physics
        mujoco.mj_step(mj_model, mj_data)
        
        # 8. Logging (every 10 steps)
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
            ik_success = True
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
        
        # 9. Print state (every 5000 steps)
        if step % 5000 == 0:
            progress = step / total_steps * 100
            print(f"Progress: {progress:.1f}% | Time: {sim_time:.2f}s | "
                  f"Target: {target_pos}")
    
    # Run simulation
    if headless:
        print(f"Simulation started (Headless). Duration: {duration}s")
        start_real_time = time.time()
        
        for step in range(total_steps):
            if not running:
                break
            step_simulation(step)
        
        elapsed = time.time() - start_real_time
        print(f"Simulation finished. Real time: {elapsed:.2f}s.")
    
    else:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            print(f"Simulation started. Duration: {duration}s")
            start_real_time = time.time()
            
            for step in range(total_steps):
                if not running:
                    break
                step_simulation(step)
                
                # Render
                if step % steps_per_render == 0:
                    viewer.sync()
            
            elapsed = time.time() - start_real_time
            print(f"Simulation finished. Real time: {elapsed:.2f}s.")


def main():
    """
    Main entry point for Delta Arm joystick simulation.
    """
    # 0. Simulation parameters
    max_sim_time = 60.0  # Maximum simulation time (seconds)
    headless = False     # Set to True for SSH/headless mode
    
    # 1. Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "meshes", "Delta_Arm.xml")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # 2. Initialize PS2 Controller
    print("\n" + "="*50)
    print("Initializing PS2 Controller")
    print("="*50)
    
    ps2_controller = PS2Controller(
        joystick_id=0,
        workspace_radius=0.10,    # XY radius: ±0.10m
        z_center=-0.15,          # Z center: 15cm below base
        z_range=0.05,            # Z range: ±5cm from center
        deadzone=0.15,           # 15% deadzone
        invert_y=True
    )
    
    try:
        ps2_controller.connect()
        ps2_controller.calibrate(samples=50)
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("  1. PS2 gamepad USB receiver is connected")
        print("  2. Gamepad is powered on")
        print("  3. Drivers are installed (usually automatic on Windows)")
        return
    
    # 3. Initialize logger
    logger = Logger()
    
    # 4. Load MuJoCo model
    print(f"\nLoading model from: {model_path}")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # 5. Create Delta model wrapper
    delta_model = DeltaModel(mj_model, mj_data)
    
    # 6. Create controller
    controller = DeltaArmController(delta_model, control_freq=100.0, control_mode='position')
    
    # 7. Initialize motor angles
    initial_motor_angles_deg = np.array([0.0, 0.0, 0.0])
    initial_motor_angles_rad = np.deg2rad(initial_motor_angles_deg)
    delta_model.set_delta_motor_positions(initial_motor_angles_rad)
    
    # 8. Run initialization for constraint convergence
    print("Initializing simulation (letting constraints converge)...")
    mujoco.mj_forward(mj_model, mj_data)
    
    for i in range(5000):
        mj_data.qvel[:] = 0
        mujoco.mj_step(mj_model, mj_data)
        
        if i % 1000 == 0:
            ee_pos = delta_model.get_ee_sensor_pos()
            print(f"  Step {i}: EE position relative to base: {ee_pos}")
    
    print("Initialization complete.")
    ee_pos_final = delta_model.get_ee_sensor_pos()
    print(f"Final EE position: {ee_pos_final}")
    
    # 9. Set initial target from joystick center
    ps2_controller.reset_position()
    
    # 10. Run simulation
    try:
        run_joystick_simulation(
            mj_model=mj_model,
            mj_data=mj_data,
            controller=controller,
            logger=logger,
            ps2_controller=ps2_controller,
            duration=max_sim_time,
            headless=headless
        )
    finally:
        # 11. Cleanup
        print("\nClosing joystick connection...")
        ps2_controller.close()
    
    # 12. Plot results
    results_path = os.path.join(current_dir, 'joystick_simulation_results.png')
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
