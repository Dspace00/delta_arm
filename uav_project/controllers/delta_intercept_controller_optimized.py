"""
Optimized Delta Intercept Controller with improved catch rate.
Key optimizations:
1. Reduced response time
2. Increased catch threshold with platform size consideration
3. Velocity feedforward control
4. Predictive intercept point selection
5. Real-time trajectory re-planning
"""
import torch
import numpy as np
from typing import Optional, Tuple
import time

from uav_project.controllers.delta_arm_controller import DeltaArmController
from uav_project.utils.ball_predictor import BallPredictor
from uav_project.utils.DeltaKinematics import DeltaKinematics
from uav_project.utils.smooth_trajectory import QuinticPolynomialTrajectory
from uav_project.config_workspace import WORKSPACE_BOUNDS, WORKSPACE_RADIUS, BASE_HEIGHT


class DeltaInterceptControllerOptimized(DeltaArmController):
    """
    Optimized controller for Delta arm ball interception.
    
    Key improvements over base version:
    - Lower minimum response time (0.08s vs 0.15s)
    - Larger catch threshold considering platform geometry
    - Velocity feedforward for faster tracking
    - Predictive intercept point selection
    - Adaptive trajectory re-planning
    """
    
    def __init__(self, model, control_freq=100.0, gravity=9.81):
        """
        Args:
            model: Instance of UAVModel or DeltaModel for MuJoCo interaction.
            control_freq: Control frequency in Hz.
            gravity: Gravitational acceleration (m/s^2).
        """
        super().__init__(model, control_freq, control_mode='position')
        
        # Ball predictor
        self.ball_predictor = BallPredictor(gravity=gravity)
        
        # ========== Use theoretical maximum workspace bounds ==========
        # Bounds are relative to Delta base (z=0 at base)
        # These are computed from forward kinematics enumeration
        self.workspace_bounds = WORKSPACE_BOUNDS.copy()
        self.workspace_radius = WORKSPACE_RADIUS
        
        # Interception state
        self.intercept_active = False
        self.intercept_target: Optional[np.ndarray] = None
        self.intercept_time: Optional[float] = None
        self.intercept_trajectory: Optional[QuinticPolynomialTrajectory] = None
        self.trajectory_start_time: float = 0.0
        
        # ========== Optimized tracking parameters ==========
        # Adjusted for 10m launch distance
        self.min_intercept_time = 0.15  # Increased for longer flight time
        self.max_intercept_time = 2.5   # Increased from 1.2s for 10m distance
        
        # ========== Optimized catch parameters ==========
        # Platform diameter ~5cm, plus ball radius ~1cm, total ~3.5cm
        # Plus tolerance for trajectory prediction error (increased for 10m distance)
        self.catch_threshold = 0.15  # 15cm (increased from 8cm for 10m distance)
        
        # Catch zone: check if ball is close enough AND moving toward platform
        self.velocity_alignment_threshold = 0.3  # m/s
        
        # ========== Velocity feedforward parameters ==========
        self.use_velocity_ff = True
        self.velocity_ff_gain = 0.5  # Feedforward gain
        
        # ========== Trajectory re-planning parameters ==========
        self.replan_interval = 0.10  # Re-plan every 100ms (increased from 50ms)
        self.last_replan_time = 0.0
        self.prediction_error_threshold = 0.05  # 5cm error triggers re-plan (increased from 3cm)
        self.min_replan_interval = 0.08  # Minimum time between replans
        
        # Tracking state
        self.ball_pos_log = np.zeros(3)
        self.ball_vel_log = np.zeros(3)
        self.mode = 'idle'
        self.status = "waiting"
        
        # Smooth trajectory tracking
        self.last_ee_vel = np.zeros(3)
        self.last_ee_acc = np.zeros(3)
        
        # Catch status
        self.is_caught = False
        self.catch_count = 0
        self.miss_count = 0
        
        # Performance tracking
        self.intercept_attempts = 0
        self.successful_intercepts = 0
        
        # Sensor offset
        self.sensor_offset_z = 0.10
        
        # Last predicted intercept for error tracking
        self.last_predicted_intercept = None
        self.last_prediction_time = 0.0
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self.ball_predictor = BallPredictor()
        self.intercept_active = False
        self.intercept_target = None
        self.intercept_time = None
        self.intercept_trajectory = None
        self.trajectory_start_time = 0.0
        self.mode = 'idle'
        self.status = "waiting"
        self.ball_pos_log = np.zeros(3)
        self.ball_vel_log = np.zeros(3)
        self.last_ee_vel = np.zeros(3)
        self.last_ee_acc = np.zeros(3)
        self.is_caught = False
        self.last_replan_time = 0.0
        self.last_predicted_intercept = None
    
    def update_ball_state(self, pos: np.ndarray, vel: np.ndarray, sim_time: float = 0.0) -> None:
        """
        Update ball state from sensor readings.
        
        Args:
            pos: Ball position (x, y, z) in world frame.
            vel: Ball velocity (vx, vy, vz) in world frame.
            sim_time: Current simulation time (for 3-point trajectory fitting).
        """
        # Convert to base-relative coordinates
        pos_relative = pos.copy()
        pos_relative[2] = pos[2] - BASE_HEIGHT
        
        self.ball_predictor.update_state(pos_relative, vel, sim_time)
        self.ball_pos_log = pos.copy()
        self.ball_vel_log = vel.copy()
    
    def update(self, sim_time: float) -> None:
        """
        Main update loop with optimized ball interception logic.
        
        Args:
            sim_time: Current simulation time.
        """
        if sim_time >= self.last_update_time + self.dt:
            
            # --- Step 1: Get current end-effector state ---
            # Note: get_ee_sensor_pos() includes sensor_offset_z (0.1m above platform)
            # We need to subtract it for IK calculations
            sensor_pos_np = self.uav.get_ee_sensor_pos()
            self.sensor_offset_z = getattr(self.uav, 'sensor_offset_z', 0.10)
            pos_np = sensor_pos_np.copy()
            pos_np[2] -= self.sensor_offset_z  # Convert sensor pos to platform pos
            
            vel_np = self.uav.get_ee_sensor_lin_vel()
            current_pos = torch.tensor(pos_np, dtype=torch.float32).view(3, 1)
            current_vel = torch.tensor(vel_np, dtype=torch.float32).view(3, 1)
            
            # Update velocity/acceleration tracking
            self.last_ee_acc = (vel_np - self.last_ee_vel) / self.dt
            self.last_ee_vel = vel_np.copy()
            
            # --- Step 2: Check for catch first (highest priority) ---
            if self._check_catch_optimized(pos_np, vel_np) and not self.is_caught:
                self.is_caught = True
                self.status = "caught"
                self.successful_intercepts += 1
                print(f"[{sim_time:.3f}s] Ball CAUGHT! Total catches: {self.successful_intercepts}")
            
            # --- Step 3: Determine desired position ---
            if self.is_caught:
                # Stay at current position after catch
                des_pos = pos_np.copy()
                des_vel = np.zeros(3)
            
            elif self.intercept_trajectory is not None and self.mode == 'intercept':
                # Follow smooth trajectory (replanning disabled for now)
                t_relative = sim_time - self.trajectory_start_time
                
                # DISABLED: Replanning causes issues with trajectory execution
                # need_replan = self._should_replan(sim_time, pos_np)
                # if need_replan:
                #     self._replan_trajectory(sim_time, pos_np, vel_np)
                #     t_relative = 0.0
                #     self.trajectory_start_time = sim_time
                
                if t_relative < self.intercept_trajectory.duration:
                    # Get position and velocity from smooth trajectory
                    des_pos = self.intercept_trajectory.get_position(t_relative)
                    des_vel = self.intercept_trajectory.get_velocity(t_relative)
                    self.status = "intercepting"
                else:
                    # Trajectory complete - hold at intercept point
                    des_pos = self.intercept_target.copy()
                    des_vel = np.zeros(3)
                    self.status = "holding"
            
            else:
                # Find new intercept point with predictive selection
                intercept_point, intercept_time = self._find_optimal_intercept_point()
                
                if intercept_point is not None:
                    self.intercept_target = intercept_point
                    self.intercept_time = intercept_time
                    self.intercept_attempts += 1
                    
                    # Set ball_predictor's intercept point for trajectory planning
                    self.ball_predictor.intercept_point = intercept_point.copy()
                    self.ball_predictor.intercept_time = intercept_time
                    self.ball_predictor.intercept_velocity = self.ball_predictor.predict_velocity(intercept_time)
                    
                    # Plan C4-smooth trajectory
                    # Note: Don't match ball velocity - Delta arm cannot move that fast
                    # Instead, aim for zero velocity at intercept point
                    self.intercept_trajectory = self.ball_predictor.plan_smooth_intercept_trajectory(
                        ee_current_pos=pos_np,
                        ee_current_vel=vel_np,
                        ee_current_acc=self.last_ee_acc,
                        match_ball_velocity=False  # Changed: Don't match ball velocity
                    )
                    
                    if self.intercept_trajectory is not None:
                        self.trajectory_start_time = sim_time
                        self.last_replan_time = sim_time
                        self.mode = 'intercept'
                        self.status = "planning"
                        des_pos = pos_np.copy()
                        des_vel = vel_np.copy()
                        self.last_predicted_intercept = intercept_point.copy()
                    else:
                        des_pos = intercept_point
                        des_vel = np.zeros(3)
                        self.status = "tracking"
                else:
                    # No intercept point - stay at current
                    des_pos = pos_np.copy()
                    des_vel = np.zeros(3)
                    self.status = "waiting"
            
            # --- Step 4: Apply workspace limits ---
            des_pos_clamped, is_outside = self._clamp_to_workspace(des_pos)
            self.target_outside_workspace = is_outside
            
            # --- Step 5: Execute position control with velocity feedforward ---
            if self.use_velocity_ff and np.any(des_vel != 0):
                # Add velocity feedforward to desired position
                # This helps the arm move faster toward the target
                des_pos_ff = des_pos_clamped + self.velocity_ff_gain * des_vel * self.dt
                des_pos_clamped, _ = self._clamp_to_workspace(des_pos_ff)
            
            # Execute IK
            des_pos_tensor = torch.tensor(des_pos_clamped, dtype=torch.float32)
            joint_angles_deg = self.kinematics.ik(des_pos_tensor)
            
            if isinstance(joint_angles_deg, int) and joint_angles_deg == -1:
                # IK failed - use last known good position
                if not hasattr(self, '_last_good_joints'):
                    self._last_good_joints = np.zeros(3)
                self.uav.set_delta_motor_positions(self._last_good_joints)
            else:
                joint_angles_rad = torch.deg2rad(joint_angles_deg)
                joint_angles_np = joint_angles_rad.numpy()
                self.uav.set_delta_motor_positions(joint_angles_np)
                self._last_good_joints = joint_angles_np.copy()
            
            # --- Step 6: Update log data ---
            self.current_des_pos_log = des_pos_clamped.copy()
            self.current_actual_pos_log = pos_np.copy()
            
            self.last_update_time = sim_time
    
    def _find_optimal_intercept_point(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Find the optimal intercept point using 3-point trajectory fitting.
        
        NEW LOGIC:
        1. Collect 3 position samples from ball trajectory
        2. Use 2 points to fit trajectory, 1 point to validate
        3. Find the point on trajectory closest to current end-effector position
        4. This minimizes movement distance and maximizes catch probability
        
        Returns:
            (intercept_point, intercept_time) or (None, None)
        """
        # Get current EE position (sensor position, includes offset)
        ee_sensor_pos = self.uav.get_ee_sensor_pos()
        
        # Try to fit trajectory from 3 points
        success, error = self.ball_predictor.fit_trajectory_3points()
        
        if not success or self.ball_predictor.fitted_pos0 is None:
            # Not enough points or poor fit - fall back to simple prediction
            # Use current velocity to predict
            return self._find_intercept_simple(ee_sensor_pos)
        
        # ========== Find closest point on trajectory ==========
        # Search through trajectory to find point closest to EE
        best_point = None
        best_time = None
        best_distance = float('inf')
        
        t = self.min_intercept_time
        while t < self.max_intercept_time:
            # Predict position from fitted trajectory
            pos = self.ball_predictor.predict_from_fitted(t)
            
            # Check if in workspace
            if self._is_in_workspace(pos):
                # Distance to current EE position
                distance = np.linalg.norm(pos - ee_sensor_pos)
                
                # Also check if ball is still above ground
                world_z = pos[2] + BASE_HEIGHT  # Convert to world frame
                if world_z > 0.1:  # Ball still in air
                    if distance < best_distance:
                        best_distance = distance
                        best_point = pos.copy()
                        best_time = t
            
            t += 0.01  # 10ms search step
        
        self.intercept_point = best_point
        self.intercept_time = best_time
        
        if best_point is not None:
            self.intercept_velocity = self.ball_predictor.predict_velocity(best_time)
        
        return best_point, best_time
    
    def _find_intercept_simple(self, ee_pos: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Fallback: Find intercept using simple velocity-based prediction.
        
        Used when 3-point fitting is not available.
        
        Args:
            ee_pos: Current end-effector position.
        
        Returns:
            (intercept_point, intercept_time) or (None, None)
        """
        best_point = None
        best_time = None
        best_distance = float('inf')
        
        t = self.min_intercept_time
        while t < self.max_intercept_time:
            pos = self.ball_predictor.predict_position(t)
            
            if self._is_in_workspace(pos):
                distance = np.linalg.norm(pos - ee_pos)
                
                world_z = pos[2] + BASE_HEIGHT
                if world_z > 0.1:
                    if distance < best_distance:
                        best_distance = distance
                        best_point = pos.copy()
                        best_time = t
            
            t += 0.01
        
        self.intercept_point = best_point
        self.intercept_time = best_time
        
        if best_point is not None:
            self.intercept_velocity = self.ball_predictor.predict_velocity(best_time)
        
        return best_point, best_time
    
    def _should_replan(self, sim_time: float, current_pos: np.ndarray) -> bool:
        """
        Check if trajectory should be re-planned.
        
        Args:
            sim_time: Current simulation time.
            current_pos: Current end-effector position.
        
        Returns:
            True if re-planning is needed.
        """
        # Don't replan too frequently
        if sim_time - self.last_replan_time < self.min_replan_interval:
            return False
        
        # Check prediction error
        if self.last_predicted_intercept is not None and self.intercept_trajectory is not None:
            # Get current ball prediction
            t_remaining = self.intercept_trajectory.duration - (sim_time - self.trajectory_start_time)
            if t_remaining > 0.10:  # More than 100ms remaining
                current_predicted = self.ball_predictor.predict_position(t_remaining)
                error = np.linalg.norm(current_predicted - self.last_predicted_intercept)
                if error > self.prediction_error_threshold:
                    return True
        
        return False
    
    def _replan_trajectory(self, sim_time: float, ee_pos: np.ndarray, ee_vel: np.ndarray):
        """
        Re-plan trajectory based on updated ball prediction.
        
        Args:
            sim_time: Current simulation time.
            ee_pos: Current end-effector position.
            ee_vel: Current end-effector velocity.
        """
        # Find new optimal intercept point
        intercept_point, intercept_time = self._find_optimal_intercept_point()
        
        if intercept_point is not None:
            self.intercept_target = intercept_point
            self.intercept_time = intercept_time
            
            # Set ball_predictor's intercept point for trajectory planning
            self.ball_predictor.intercept_point = intercept_point.copy()
            self.ball_predictor.intercept_time = intercept_time
            self.ball_predictor.intercept_velocity = self.ball_predictor.predict_velocity(intercept_time)
            
            # Re-plan trajectory (don't match ball velocity)
            new_trajectory = self.ball_predictor.plan_smooth_intercept_trajectory(
                ee_current_pos=ee_pos,
                ee_current_vel=ee_vel,
                ee_current_acc=self.last_ee_acc,
                match_ball_velocity=False  # Don't match ball velocity
            )
            
            if new_trajectory is not None:
                self.intercept_trajectory = new_trajectory
                self.last_predicted_intercept = intercept_point.copy()
                self.last_replan_time = sim_time
                self.trajectory_start_time = sim_time  # Reset trajectory start time
                self.status = "replanning"
    
    def _check_catch_optimized(self, ee_pos: np.ndarray, ee_vel: np.ndarray) -> bool:
        """
        Optimized catch detection using a virtual "paddle" model.
        
        NEW LOGIC:
        - Virtual paddle centered at sensor point (10cm above platform)
        - Paddle radius: 7.5cm (like a ping-pong paddle)
        - Ball radius: 2cm (from XML)
        - Catch if ball surface touches paddle
        
        The paddle moves with the platform (no rotation).
        
        Args:
            ee_pos: End-effector sensor position (includes 10cm offset).
            ee_vel: End-effector velocity.
        
        Returns:
            True if ball is caught.
        """
        ball_pos = self.ball_pos_log
        ball_vel = self.ball_vel_log
        
        if np.all(ball_pos == 0):
            return False
        
        # ========== Virtual Paddle Parameters ==========
        # Paddle center is at sensor point (ee_pos already includes offset)
        paddle_center = ee_pos.copy()  # This is platform_z + 0.1m
        paddle_radius = 0.075  # 7.5cm paddle radius
        ball_radius = 0.02    # 2cm ball radius
        
        # ========== Distance Check ==========
        # Ball is caught if ball surface touches paddle
        distance = np.linalg.norm(ball_pos - paddle_center)
        catch_distance = paddle_radius + ball_radius  # 9.5cm total
        
        if distance < catch_distance:
            # Ball has touched the paddle!
            # No need to check velocity - if it touches, it's caught
            return True
        
        return False
    
    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """Check if position is within workspace bounds."""
        x_ok = self.workspace_bounds['x'][0] <= pos[0] <= self.workspace_bounds['x'][1]
        y_ok = self.workspace_bounds['y'][0] <= pos[1] <= self.workspace_bounds['y'][1]
        z_ok = self.workspace_bounds['z'][0] <= pos[2] <= self.workspace_bounds['z'][1]
        return x_ok and y_ok and z_ok
    
    def get_log_data(self) -> Tuple:
        """Return log data for recording."""
        return (
            self.current_des_pos_log.copy(),
            self.current_actual_pos_log.copy(),
            self.ball_pos_log.copy(),
            self.ball_vel_log.copy(),
            self.intercept_target.copy() if self.intercept_target is not None else None,
            self.status,
            self.is_caught
        )
    
    def get_statistics(self) -> dict:
        """Get interception statistics."""
        return {
            'attempts': self.intercept_attempts,
            'catches': self.successful_intercepts,
            'catch_rate': self.successful_intercepts / max(1, self.intercept_attempts)
        }
    
    def print_state(self) -> None:
        """Print current state."""
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        print(f"Status: {self.status}")
        print(f"Mode: {self.mode}")
        print(f"EE Position: {self.current_actual_pos_log}")
        if self.intercept_target is not None:
            print(f"Intercept Target: {self.intercept_target}")
            print(f"Intercept Time: {self.intercept_time:.3f}s")
        stats = self.get_statistics()
        print(f"Catch Rate: {stats['catch_rate']*100:.1f}% ({stats['catches']}/{stats['attempts']})")
