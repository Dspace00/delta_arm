"""
Main entry point for the UAV simulation project.
Run this from the parent directory using: python -m uav_project.main
"""
import os
import sys
import mujoco

# Add the project root to sys.path to allow running this script directly
# This ensures that 'import uav_project.xxx' works even if we are inside the folder
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Adjust imports to work when run as a module
from uav_project.simulation.simulator import Simulator
# from uav_project.controllers.cascade_controller import CascadeController
from uav_project.controllers.combined_controller import CombinedController
from uav_project.models.uav_model import UAVModel
from uav_project.utils.logger import Logger
from uav_project.utils.trajectory import generate_circular_trajectory, generate_spiral_trajectory

def main():
    # 0. Set total sim time (in seconds)
    total_sim_time = 100.0

    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Delta_project
    
    # You can switch between "Delta.xml" and "UAV.xml" here
    model_filename = "Delta.xml" 
    model_path = os.path.join(current_dir, "meshes", model_filename)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # 2. Initialize Logger
    logger = Logger()
    
    # 3. Initialize Simulator (loads model)
    # import mujoco
    print(f"Loading model from: {model_path}")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # 4. Initialize Components
    simulator = Simulator(model_path, None, logger) # Pass None for controller initially
    
    # Use the Simulator's model/data for the Controller
    uav_model = UAVModel(simulator.model, simulator.data)
    
    # Use CombinedController to control both UAV and Delta
    controller = CombinedController(uav_model)
    
    # Set the controller back to the simulator
    simulator.controller = controller

    # 6. Define Trajectory
    trajectory = generate_circular_trajectory(
        center=[0, 0, 1.0],
        radius=2.0,
        total_time=total_sim_time,
        num_points=100,
        clockwise=False,
        height_variation=False
    )
    trajectory0 = [
        (0.0, [0, 0, 1.5]),
        (100.0, [0, 0, 1.5])
    ]
    trajectory1 = generate_spiral_trajectory(
        center=[0, 0, 1.0],
        start_radius=1.0,
        end_radius=2.0,
        total_time=total_sim_time,
        num_turns=3,
        num_points=100,
        clockwise=False
    )

    # 7. Run Simulation (Headless=True due to SSH constraints)
    simulator.run(duration=total_sim_time, trajectory=trajectory0, headless=False)
    
    # 8. Plot Results
    logger.plot_results(save_path=os.path.join(current_dir, 'simulation_results.png'))

if __name__ == "__main__":
    main()
