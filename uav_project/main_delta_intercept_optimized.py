"""
Optimized Delta arm multi-ball interception simulation.
Features:
1. Higher catch rate through optimized parameters
2. Optional visualization (headless mode can be disabled)
3. Detailed statistics and analysis
4. Adaptive difficulty progression

Run: python -m uav_project.main_delta_intercept_optimized
"""
import os
import sys
import time
import mujoco
import numpy as np
import mujoco.viewer
from typing import Tuple, Optional, List

# Path setup
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from uav_project.models.delta_ball_model import DeltaBallModel
from uav_project.controllers.delta_intercept_controller_optimized import DeltaInterceptControllerOptimized
from uav_project.utils.logger import Logger
from uav_project.utils.ball_trajectory_generator import OptimizedBallTrajectoryGenerator, TrajectoryDifficultyAnalyzer
from uav_project.config_workspace import WORKSPACE_BOUNDS, WORKSPACE_RADIUS, BASE_HEIGHT


def run_optimized_simulation(
    n_balls: int = 100,
    headless: bool = False,  # Default to showing visualization
    difficulty_progression: bool = True,
    verbose: bool = True
):
    """
    Run optimized continuous ball interception simulation.
    
    Args:
        n_balls: Number of balls to throw.
        headless: If False, shows MuJoCo visualization window.
        difficulty_progression: If True, difficulty increases over time.
        verbose: If True, prints detailed progress.
    """
    print("=" * 70)
    print("OPTIMIZED Delta Arm Multi-Ball Interception Simulation")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Balls: {n_balls}")
    print(f"  - Headless: {headless}")
    print(f"  - Difficulty progression: {difficulty_progression}")
    print(f"  - Optimized parameters: enabled")
    print("=" * 70)
    
    # ========== Load Model ==========
    model_path = os.path.join(current_file_dir, "meshes", "Delta_Ball.xml")
    print(f"\nLoading model from: {model_path}")
    
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # ========== Initialize Components ==========
    delta_model = DeltaBallModel(mj_model, mj_data, sensor_offset_z=0.10)
    controller = DeltaInterceptControllerOptimized(delta_model, control_freq=100.0)
    logger = Logger()
    trajectory_gen = OptimizedBallTrajectoryGenerator()  # Uses BASE_HEIGHT from config
    difficulty_analyzer = TrajectoryDifficultyAnalyzer()
    
    print(f"Base height: {BASE_HEIGHT}m")
    print(f"Workspace Z (world): {BASE_HEIGHT + WORKSPACE_BOUNDS['z'][0]:.3f} - {BASE_HEIGHT + WORKSPACE_BOUNDS['z'][1]:.3f}m")
    
    # ========== Initial Convergence ==========
    print("\nInitializing Delta arm...")
    
    # Use -45 degrees initial angle to position platform within workspace
    initial_angles = np.deg2rad(np.array([-45.0, -45.0, -45.0]))
    delta_model.set_delta_motor_positions(initial_angles)
    mujoco.mj_forward(mj_model, mj_data)
    
    for i in range(3000):
        mj_data.qvel[:] = 0
        mujoco.mj_step(mj_model, mj_data)
    
    ee_pos = delta_model.get_ee_sensor_pos()
    ee_world = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + BASE_HEIGHT])
    print(f"Initial EE position (world): {ee_world}")
    
    # ========== Generate All Trajectories ==========
    print("\nGenerating ball trajectories...")
    trajectories = trajectory_gen.generate_batch(
        n_balls=n_balls,
        ee_start_pos=ee_world,
        difficulty_progression=difficulty_progression
    )
    
    # Analyze difficulty distribution
    difficulties = []
    for start_pos, start_vel, intercept_pt, flight_time, ball_id in trajectories:
        diff = difficulty_analyzer.calculate_difficulty(
            intercept_pt, flight_time, ee_world
        )
        difficulties.append(diff)
    
    print(f"Difficulty distribution:")
    print(f"  - Easy (< 0.3): {sum(1 for d in difficulties if d < 0.3)}")
    print(f"  - Medium (0.3-0.6): {sum(1 for d in difficulties if 0.3 <= d < 0.6)}")
    print(f"  - Hard (> 0.6): {sum(1 for d in difficulties if d >= 0.6)}")
    
    # ========== Simulation Parameters ==========
    timestep = mj_model.opt.timestep
    max_time_per_ball = 1.2  # Reduced from 1.5s
    ball_throw_delay = 0.03  # Reduced from 0.05s
    
    # ========== Results Tracking ==========
    results = {
        'ball_ids': [],
        'caught': [],
        'flight_times': [],
        'difficulties': [],
        'catch_times': [],
        'intercept_points': [],
        'ball_trajectories': [],
        'ee_trajectories': []
    }
    
    # ========== Run Simulation ==========
    print("\n" + "=" * 70)
    print("Starting simulation...")
    print("-" * 70)
    
    start_time = time.time()
    sim_time = 0.0
    
    current_ball_idx = 0
    ball_in_flight = False
    ball_caught = False
    ball_throw_time = 0.0
    ball_trajectory = []
    ee_trajectory = []
    
    last_ee_world = ee_world.copy()
    
    total_sim_steps = int((n_balls * max_time_per_ball + 5.0) / timestep)
    
    if headless:
        # Headless mode - no visualization
        step = 0
        while current_ball_idx < n_balls and step < total_sim_steps:
            sim_time = step * timestep
            
            # Ball throwing logic
            if not ball_in_flight and current_ball_idx < n_balls:
                start_pos, start_vel, intercept_pt, flight_time, ball_id = \
                    trajectories[current_ball_idx]
                
                # Set ball state using DeltaBallModel method
                delta_model.set_ball_state(start_pos, start_vel)
                
                ball_in_flight = True
                ball_caught = False
                ball_throw_time = sim_time
                ball_trajectory = []
                ee_trajectory = []
                
                controller.is_caught = False
                controller.mode = 'idle'
                controller.intercept_trajectory = None
                
                if verbose:
                    diff = difficulties[current_ball_idx]
                    print(f"[{sim_time:.2f}s] Ball {current_ball_idx} thrown "
                          f"(difficulty: {diff:.2f}, flight time: {flight_time:.2f}s)")
            
            # Update ball state to controller
            if ball_in_flight:
                ball_pos = delta_model.get_ball_pos()
                ball_vel = delta_model.get_ball_vel()
                controller.update_ball_state(ball_pos, ball_vel, sim_time)
                
                ball_trajectory.append(ball_pos.copy())
                ee_trajectory.append(last_ee_world.copy())
                
                # Check if ball caught
                if controller.is_caught and not ball_caught:
                    ball_caught = True
                
                # Check if ball simulation complete
                elapsed = sim_time - ball_throw_time
                if elapsed > max_time_per_ball or ball_pos[2] < 0.05 or ball_caught:
                    # Record result
                    results['ball_ids'].append(current_ball_idx)
                    results['caught'].append(ball_caught)
                    results['flight_times'].append(
                        trajectories[current_ball_idx][3]
                    )
                    results['difficulties'].append(difficulties[current_ball_idx])
                    results['catch_times'].append(elapsed if ball_caught else -1)
                    results['intercept_points'].append(
                        trajectories[current_ball_idx][2]
                    )
                    results['ball_trajectories'].append(ball_trajectory.copy())
                    results['ee_trajectories'].append(ee_trajectory.copy())
                    
                    if verbose and ball_caught:
                        print(f"[{sim_time:.2f}s] Ball {current_ball_idx} CAUGHT!")
                    
                    # Move to next ball
                    current_ball_idx += 1
                    ball_in_flight = False
                    
                    # Update last EE position
                    last_ee_world = np.array([
                        ee_trajectory[-1][0] if ee_trajectory else 0,
                        ee_trajectory[-1][1] if ee_trajectory else 0,
                        ee_trajectory[-1][2] if ee_trajectory else 0.4
                    ])
            
            # Controller update
            controller.update(sim_time)
            
            # Physics step
            mj_data.qvel[:] *= 0.999  # Small damping
            mujoco.mj_step(mj_model, mj_data)
            
            # Update last EE position
            ee_pos = delta_model.get_ee_sensor_pos()
            last_ee_world = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + BASE_HEIGHT])
            
            step += 1
            
            # Progress update
            if step % 10000 == 0 and verbose:
                n_caught = sum(results['caught'])
                print(f"Progress: {current_ball_idx}/{n_balls} balls, "
                      f"{n_caught} caught ({100*n_caught/max(1,current_ball_idx):.1f}%)")
    
    else:
        # Visualization mode
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            step = 0
            while current_ball_idx < n_balls and step < total_sim_steps and viewer.is_running():
                sim_time = step * timestep
                
                # Ball throwing logic (same as above)
                if not ball_in_flight and current_ball_idx < n_balls:
                    start_pos, start_vel, intercept_pt, flight_time, ball_id = \
                        trajectories[current_ball_idx]
                    
                    # Set ball state using DeltaBallModel method
                    delta_model.set_ball_state(start_pos, start_vel)
                    
                    ball_in_flight = True
                    ball_caught = False
                    ball_throw_time = sim_time
                    ball_trajectory = []
                    ee_trajectory = []
                    
                    controller.is_caught = False
                    controller.mode = 'idle'
                    controller.intercept_trajectory = None
                
                # Update ball state
                if ball_in_flight:
                    ball_pos = delta_model.get_ball_pos()
                    ball_vel = delta_model.get_ball_vel()
                    controller.update_ball_state(ball_pos, ball_vel, sim_time)
                    
                    ball_trajectory.append(ball_pos.copy())
                    ee_trajectory.append(last_ee_world.copy())
                    
                    if controller.is_caught and not ball_caught:
                        ball_caught = True
                    
                    elapsed = sim_time - ball_throw_time
                    if elapsed > max_time_per_ball or ball_pos[2] < 0.05 or ball_caught:
                        results['ball_ids'].append(current_ball_idx)
                        results['caught'].append(ball_caught)
                        results['flight_times'].append(trajectories[current_ball_idx][3])
                        results['difficulties'].append(difficulties[current_ball_idx])
                        results['catch_times'].append(elapsed if ball_caught else -1)
                        results['intercept_points'].append(trajectories[current_ball_idx][2])
                        results['ball_trajectories'].append(ball_trajectory.copy())
                        results['ee_trajectories'].append(ee_trajectory.copy())
                        
                        current_ball_idx += 1
                        ball_in_flight = False
                        
                        ee_pos = delta_model.get_ee_sensor_pos()
                        last_ee_world = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + BASE_HEIGHT])
                
                # Controller update
                controller.update(sim_time)
                
                # Physics step
                mj_data.qvel[:] *= 0.999
                mujoco.mj_step(mj_model, mj_data)
                
                ee_pos = delta_model.get_ee_sensor_pos()
                last_ee_world = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + BASE_HEIGHT])
                
                # Render
                if step % 10 == 0:
                    viewer.sync()
                
                step += 1
    
    # ========== Results Summary ==========
    total_time = time.time() - start_time
    n_caught = sum(results['caught'])
    n_missed = len(results['caught']) - n_caught
    catch_rate = n_caught / len(results['caught']) if results['caught'] else 0
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Total balls:  {len(results['caught'])}")
    print(f"Caught:       {n_caught} ({100*catch_rate:.1f}%)")
    print(f"Missed:       {n_missed} ({100*(1-catch_rate):.1f}%)")
    print(f"Sim time:     {total_time:.2f}s")
    
    # Difficulty breakdown
    if results['caught']:
        easy_caught = sum(1 for i, c in enumerate(results['caught']) 
                         if c and results['difficulties'][i] < 0.3)
        medium_caught = sum(1 for i, c in enumerate(results['caught']) 
                           if c and 0.3 <= results['difficulties'][i] < 0.6)
        hard_caught = sum(1 for i, c in enumerate(results['caught']) 
                         if c and results['difficulties'][i] >= 0.6)
        
        easy_total = sum(1 for d in results['difficulties'] if d < 0.3)
        medium_total = sum(1 for d in results['difficulties'] if 0.3 <= d < 0.6)
        hard_total = sum(1 for d in results['difficulties'] if d >= 0.6)
        
        print(f"\nCatch rate by difficulty:")
        if easy_total > 0:
            print(f"  Easy:   {easy_caught}/{easy_total} ({100*easy_caught/easy_total:.1f}%)")
        if medium_total > 0:
            print(f"  Medium: {medium_caught}/{medium_total} ({100*medium_caught/medium_total:.1f}%)")
        if hard_total > 0:
            print(f"  Hard:   {hard_caught}/{hard_total} ({100*hard_caught/hard_total:.1f}%)")
    
    print("=" * 70)
    
    # ========== Save Results ==========
    # Log to Logger for visualization
    for i, (ball_id, caught, ball_traj, ee_traj) in enumerate(zip(
            results['ball_ids'], results['caught'], 
            results['ball_trajectories'], results['ee_trajectories'])):
        
        catch_time = results['catch_times'][i]
        intercept_pt = results['intercept_points'][i]
        
        logger.log_interception(
            ball_id=ball_id,
            ball_trajectory=ball_traj,
            ee_trajectory=ee_traj,
            catch_time=catch_time if caught else -1,
            is_caught=caught,
            intercept_point=intercept_pt
        )
    
    # Generate plots
    save_path = os.path.join(current_file_dir, 'optimized_interception_results.png')
    logger.plot_multi_interception_results(save_path=save_path)
    
    # Save example plots
    caught_indices = [i for i, c in enumerate(results['caught']) if c]
    missed_indices = [i for i, c in enumerate(results['caught']) if not c]
    
    if caught_indices:
        example_idx = caught_indices[0]
        logger.plot_single_interception_3d(
            results['ball_ids'][example_idx],
            save_path=os.path.join(current_file_dir, 'optimized_caught_example.png')
        )
    
    if missed_indices:
        example_idx = missed_indices[0]
        logger.plot_single_interception_3d(
            results['ball_ids'][example_idx],
            save_path=os.path.join(current_file_dir, 'optimized_missed_example.png')
        )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Delta Interception Simulation')
    parser.add_argument('--n_balls', type=int, default=100, help='Number of balls')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--no_difficulty_progression', action='store_true', 
                       help='Disable difficulty progression')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    results = run_optimized_simulation(
        n_balls=args.n_balls,
        headless=args.headless,
        difficulty_progression=not args.no_difficulty_progression,
        verbose=not args.quiet
    )
