"""
Main entry point for Delta arm multi-ball interception simulation.
Continuous mode: 100 balls thrown sequentially, Delta arm never resets.

Run from parent directory: python -m uav_project.main_delta_intercept

Key Features:
1. 100 balls, thrown continuously with minimal delay
2. Delta arm stays in position between catches (no reset)
3. Ball trajectories pass through Delta workspace + gravity constraint
4. C4-smooth trajectory planning (0~4th derivatives smooth)
5. Data collection at 10cm above platform center
"""
import os
import sys
import time
import mujoco
import numpy as np
from typing import Tuple, Optional

# Path setup
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from uav_project.models.delta_ball_model import DeltaBallModel
from uav_project.controllers.delta_intercept_controller import DeltaInterceptController
from uav_project.utils.logger import Logger


class ContinuousBallThrower:
    """
    Manages continuous ball throwing with proper timing.
    Ensures balls enter Delta workspace and are thrown at appropriate intervals.
    """
    
    def __init__(self, base_height: float = 0.5, workspace_radius: float = 0.10,
                 workspace_z_min: float = 0.30, workspace_z_max: float = 0.45):
        """
        Args:
            base_height: Height of Delta base above ground.
            workspace_radius: Radius of Delta workspace.
            workspace_z_min: Minimum Z of workspace (world frame).
            workspace_z_max: Maximum Z of workspace (world frame).
        """
        self.base_height = base_height
        self.workspace_radius = workspace_radius
        self.workspace_z_min = workspace_z_min
        self.workspace_z_max = workspace_z_max
        self.g = 9.81
    
    def generate_ball_params(self, ball_id: int, ee_current_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ball initial position and velocity.
        
        Ball is thrown such that it passes through Delta workspace.
        The intercept point is chosen to be reachable from ee_current_pos.
        
        Args:
            ball_id: Ball identifier.
            ee_current_pos: Current end-effector position (world frame).
        
        Returns:
            (start_pos, start_vel) tuple.
        """
        np.random.seed(42 + ball_id)
        
        # Choose intercept point near current EE position for smooth transition
        # This reduces the distance the arm needs to travel
        
        # Extract current EE position
        ee_x, ee_y, ee_z = ee_current_pos
        
        # Bias intercept point toward current EE position
        # But keep within workspace bounds
        intercept_x = np.clip(
            ee_x + np.random.uniform(-0.05, 0.05),
            -self.workspace_radius * 0.9,
            self.workspace_radius * 0.9
        )
        intercept_y = np.clip(
            ee_y + np.random.uniform(-0.05, 0.05),
            -self.workspace_radius * 0.9,
            self.workspace_radius * 0.9
        )
        intercept_z = np.random.uniform(self.workspace_z_min, self.workspace_z_max)
        
        # Flight time (time for ball to reach intercept point)
        flight_time = np.random.uniform(0.25, 0.6)
        
        # Initial height (above intercept point)
        z0 = np.random.uniform(intercept_z + 0.2, intercept_z + 0.8)
        
        # Calculate required vertical velocity
        vz0 = (intercept_z - z0 + 0.5 * self.g * flight_time**2) / flight_time
        
        # Random start position (ball launcher position)
        start_x = np.random.uniform(-0.20, 0.20)
        start_y = np.random.uniform(-0.20, 0.20)
        
        # Adjust velocities to reach intercept point from start
        vx0 = (intercept_x - start_x) / flight_time
        vy0 = (intercept_y - start_y) / flight_time
        
        start_pos = np.array([start_x, start_y, z0])
        start_vel = np.array([vx0, vy0, vz0])
        
        return start_pos, start_vel
    
    def estimate_arrival_time(self, start_pos: np.ndarray, start_vel: np.ndarray,
                              intercept_z: float) -> float:
        """
        Estimate when ball will reach target Z height.
        
        Args:
            start_pos: Ball start position.
            start_vel: Ball start velocity.
            intercept_z: Target Z height.
        
        Returns:
            Estimated arrival time.
        """
        # z(t) = z0 + vz0*t - 0.5*g*t^2
        # Solve: 0.5*g*t^2 - vz0*t + (intercept_z - z0) = 0
        a = 0.5 * self.g
        b = -start_vel[2]
        c = intercept_z - start_pos[2]
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return 1.0  # Default
        
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Return the positive time
        times = [t for t in [t1, t2] if t > 0]
        return min(times) if times else 1.0


def run_continuous_simulation(
    n_balls: int = 100,
    headless: bool = True
):
    """
    Run continuous ball interception simulation.
    
    Key features:
    - Balls thrown sequentially with minimal delay
    - Delta arm never resets between balls
    - Path planning starts from previous endpoint
    
    Args:
        n_balls: Number of balls to throw.
        headless: Whether to run without visualization.
    """
    print("=" * 70)
    print("Delta Arm Continuous Multi-Ball Interception Simulation")
    print("=" * 70)
    print(f"Configuration: {n_balls} balls, continuous mode, no arm reset")
    print(f"Trajectory: C4-smooth (quintic polynomial)")
    print(f"Data collection: 10cm above platform center")
    print("=" * 70)
    
    # ========== Load Model ==========
    model_path = os.path.join(current_file_dir, "meshes", "Delta_Ball.xml")
    print(f"\nLoading model from: {model_path}")
    
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # ========== Initialize Model Wrapper ==========
    delta_model = DeltaBallModel(
        mj_model, mj_data,
        sensor_offset_z=0.10
    )
    
    # ========== Initialize Controller ==========
    controller = DeltaInterceptController(delta_model, control_freq=100.0)
    
    # ========== Initialize Logger ==========
    logger = Logger()
    
    # ========== Ball Thrower ==========
    thrower = ContinuousBallThrower(
        base_height=0.5,
        workspace_radius=0.10,
        workspace_z_min=0.30,
        workspace_z_max=0.45
    )
    
    # ========== Initial Convergence ==========
    print("\nInitializing Delta arm...")
    
    initial_angles = np.array([0.0, 0.0, 0.0])
    delta_model.set_delta_motor_positions(initial_angles)
    mujoco.mj_forward(mj_model, mj_data)
    
    for i in range(3000):
        mj_data.qvel[:] = 0
        mujoco.mj_step(mj_model, mj_data)
    
    ee_pos = delta_model.get_ee_sensor_pos()
    ee_world = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + 0.5])
    print(f"Initial EE position (world): {ee_world}")
    print("Initialization complete.\n")
    
    # ========== Simulation Parameters ==========
    timestep = mj_model.opt.timestep
    
    # Maximum time per ball (if not caught, move to next)
    max_time_per_ball = 1.5
    
    # Delay between balls (in simulation time)
    ball_throw_delay = 0.05  # 50ms - very short delay
    
    # ========== Run Continuous Simulation ==========
    print(f"Starting continuous {n_balls}-ball simulation...")
    print("-" * 70)
    
    start_time = time.time()
    sim_time = 0.0
    
    results = []
    
    # Ball state tracking
    current_ball_id = 0
    ball_in_flight = False
    ball_caught = False
    ball_missed = False
    ball_throw_time = 0.0
    ball_trajectory = []
    ee_trajectory = []
    
    # Last EE position (for next ball's trajectory planning)
    last_ee_world = ee_world.copy()
    
    total_sim_steps = int((n_balls * max_time_per_ball + 5.0) / timestep)
    
    step = 0
    while current_ball_id < n_balls and step < total_sim_steps:
        sim_time = step * timestep
        
        # Check if we need to throw a new ball
        if not ball_in_flight and current_ball_id < n_balls:
            # Throw new ball
            start_pos, start_vel = thrower.generate_ball_params(
                ball_id=current_ball_id,
                ee_current_pos=last_ee_world
            )
            
            delta_model.set_ball_state(start_pos, start_vel)
            ball_in_flight = True
            ball_caught = False
            ball_missed = False
            ball_throw_time = sim_time
            ball_trajectory = []
            ee_trajectory = []
            
            # Reset controller for new ball (but keep position!)
            controller.reset()
            controller.update_ball_state(start_pos, start_vel)
            
            if current_ball_id % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Ball {current_ball_id:3d}/{n_balls} | Sim time: {sim_time:.2f}s | Elapsed: {elapsed:.1f}s")
        
        # Read ball state
        ball_pos = delta_model.get_ball_pos()
        ball_vel = delta_model.get_ball_vel()
        
        # Update controller
        controller.update_ball_state(ball_pos, ball_vel)
        controller.update(sim_time)
        
        # Physics step
        mujoco.mj_step(mj_model, mj_data)
        
        # Log data (every 20 steps)
        if step % 20 == 0:
            ee_pos = delta_model.get_ee_sensor_pos()
            ee_world = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + 0.5])
            ball_trajectory.append(ball_pos.copy())
            ee_trajectory.append(ee_world.copy())
            last_ee_world = ee_world.copy()
        
        # Check for catch
        if ball_in_flight and controller.is_caught and not ball_caught:
            ball_caught = True
            ball_in_flight = False
            print(f"[Ball {current_ball_id:3d}] CAUGHT at t={sim_time:.3f}s")
            
            # Log result
            logger.log_interception(
                ball_id=current_ball_id,
                ball_trajectory=ball_trajectory,
                ee_trajectory=ee_trajectory,
                catch_time=sim_time - ball_throw_time,
                is_caught=True,
                intercept_point=controller.intercept_target.copy() if controller.intercept_target is not None else np.zeros(3)
            )
            
            results.append({
                'ball_id': current_ball_id,
                'is_caught': True,
                'catch_time': sim_time - ball_throw_time
            })
            
            current_ball_id += 1
        
        # Check for miss (ball hit ground or too long)
        time_since_throw = sim_time - ball_throw_time
        if ball_in_flight and (ball_pos[2] < 0.02 or time_since_throw > max_time_per_ball):
            ball_missed = True
            ball_in_flight = False
            
            if ball_pos[2] < 0.02:
                # Ball hit ground - let it stay there
                delta_model.reset_ball_on_ground()
            
            # Log result
            logger.log_interception(
                ball_id=current_ball_id,
                ball_trajectory=ball_trajectory,
                ee_trajectory=ee_trajectory,
                catch_time=-1.0,
                is_caught=False,
                intercept_point=controller.intercept_target.copy() if controller.intercept_target is not None else np.zeros(3)
            )
            
            results.append({
                'ball_id': current_ball_id,
                'is_caught': False,
                'catch_time': -1.0
            })
            
            current_ball_id += 1
        
        step += 1
    
    total_time = time.time() - start_time
    
    # ========== Print Summary ==========
    n_caught = sum(r['is_caught'] for r in results)
    n_missed = len(results) - n_caught
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Total balls:  {len(results)}")
    print(f"Caught:       {n_caught} ({100*n_caught/max(len(results),1):.1f}%)")
    print(f"Missed:       {n_missed} ({100*n_missed/max(len(results),1):.1f}%)")
    print(f"Total time:   {total_time:.1f}s")
    print(f"Sim time:     {sim_time:.2f}s")
    
    # Catch time statistics
    catch_times = [r['catch_time'] for r in results if r['is_caught']]
    if catch_times:
        print(f"\nCatch time - Mean: {np.mean(catch_times):.3f}s")
        print(f"Catch time - Std:  {np.std(catch_times):.3f}s")
        print(f"Catch time - Min:  {np.min(catch_times):.3f}s")
        print(f"Catch time - Max:  {np.max(catch_times):.3f}s")
    
    print("=" * 70)
    
    # ========== Generate Plots ==========
    print("\nGenerating visualization...")
    
    logger.plot_multi_interception_results(
        save_path=os.path.join(current_file_dir, 'continuous_interception_results.png')
    )
    
    # Plot examples
    caught_ids = [r['ball_id'] for r in results if r['is_caught']]
    missed_ids = [r['ball_id'] for r in results if not r['is_caught']]
    
    if caught_ids:
        logger.plot_single_interception_3d(
            ball_id=caught_ids[0],
            save_path=os.path.join(current_file_dir, 'continuous_caught_example.png')
        )
    
    if missed_ids:
        logger.plot_single_interception_3d(
            ball_id=missed_ids[0],
            save_path=os.path.join(current_file_dir, 'continuous_missed_example.png')
        )
    
    print("\n" + "=" * 70)
    print("All results saved!")
    print("=" * 70)
    
    return results


def main():
    """Main entry point."""
    run_continuous_simulation(n_balls=100, headless=True)


if __name__ == "__main__":
    main()
