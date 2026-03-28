"""
Delta Intercept Controller for ball tracking simulation.
Integrates ball prediction with Delta arm control to intercept flying ball.
Uses C4-smooth trajectory planning (quintic polynomial).
"""
import torch
import numpy as np
from typing import Optional, Tuple

from uav_project.controllers.delta_arm_controller import DeltaArmController
from uav_project.utils.ball_predictor import BallPredictor
from uav_project.utils.DeltaKinematics import DeltaKinematics
from uav_project.utils.smooth_trajectory import QuinticPolynomialTrajectory


class DeltaInterceptController(DeltaArmController):
    """
    Controller for Delta arm to intercept flying ball.
    
    Extends DeltaArmController with:
    - Ball trajectory prediction
    - Intercept point calculation
    - C4-smooth trajectory planning (quintic polynomial)
    - Smooth motion with continuous 0~4th derivatives
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
        
        # Delta-specific workspace bounds (relative to base)
        self.workspace_bounds = {
            'x': (-0.10, 0.10),
            'y': (-0.10, 0.10),
            'z': (-0.20, -0.05)  # Intercept zone: 5-20cm below base
        }
        
        # Interception state
        self.intercept_active = False
        self.intercept_target: Optional[np.ndarray] = None
        self.intercept_time: Optional[float] = None
        self.intercept_trajectory: Optional[QuinticPolynomialTrajectory] = None
        self.trajectory_start_time: float = 0.0
        
        # Tracking state
        self.ball_pos_log = np.zeros(3)
        self.ball_vel_log = np.zeros(3)
        
        # Mode: 'idle', 'track', 'intercept'
        self.mode = 'idle'
        
        # Tracking parameters
        self.min_intercept_time = 0.15  # Minimum time before intercept
        
        # Status
        self.status = "waiting"
        
        # Smooth trajectory tracking
        self.last_ee_vel = np.zeros(3)
        self.last_ee_acc = np.zeros(3)
        
        # Catch detection
        self.catch_threshold = 0.05  # 5cm for end-effector platform center + 10cm offset
        self.is_caught = False
        
        # Sensor offset: measure point is 10cm above platform center
        # User requirement: data collected at a point 10cm above platform center
        self.sensor_offset_z = 0.10  # 10cm
    
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
    
    def update_ball_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """
        Update ball state from sensor readings.
        
        Args:
            pos: Ball position (x, y, z) in world frame.
            vel: Ball velocity (vx, vy, vz) in world frame.
        """
        # Convert to base-relative coordinates
        # Base is at Z=0.5 in world frame
        pos_relative = pos.copy()
        pos_relative[2] = pos[2] - 0.5  # Z relative to base
        
        self.ball_predictor.update_state(pos_relative, vel)
        self.ball_pos_log = pos.copy()
        self.ball_vel_log = vel.copy()
    
    def update(self, sim_time: float) -> None:
        """
        Main update loop with ball interception logic.
        Uses C4-smooth trajectory planning.
        
        Args:
            sim_time: Current simulation time.
        """
        if sim_time >= self.last_update_time + self.dt:
            
            # --- Step 1: Get current end-effector state ---
            pos_np = self.uav.get_ee_sensor_pos()
            vel_np = self.uav.get_ee_sensor_lin_vel()
            current_pos = torch.tensor(pos_np, dtype=torch.float32).view(3, 1)
            current_vel = torch.tensor(vel_np, dtype=torch.float32).view(3, 1)
            
            # Update velocity/acceleration tracking
            self.last_ee_acc = (vel_np - self.last_ee_vel) / self.dt
            self.last_ee_vel = vel_np.copy()
            
            # --- Step 2: Determine desired position ---
            if self.is_caught:
                # Stay at current position after catch
                des_pos = pos_np.copy()
                self.status = "caught"
            
            elif self.intercept_trajectory is not None and self.mode == 'intercept':
                # Follow smooth trajectory
                t_relative = sim_time - self.trajectory_start_time
                
                if t_relative < self.intercept_trajectory.duration:
                    # Get position from smooth trajectory
                    des_pos = self.intercept_trajectory.get_position(t_relative)
                    self.status = "intercepting"
                else:
                    # Trajectory complete - hold at intercept point
                    des_pos = self.intercept_target.copy()
                    self.status = "holding"
            
            else:
                # Find new intercept point
                intercept_point, intercept_time = self.ball_predictor.find_intercept_point(
                    workspace_bounds=self.workspace_bounds,
                    min_reaction_time=self.min_intercept_time,
                    max_intercept_time=1.5
                )
                
                if intercept_point is not None:
                    self.intercept_target = intercept_point
                    self.intercept_time = intercept_time
                    
                    # Plan C4-smooth trajectory
                    self.intercept_trajectory = self.ball_predictor.plan_smooth_intercept_trajectory(
                        ee_current_pos=pos_np,
                        ee_current_vel=vel_np,
                        ee_current_acc=self.last_ee_acc,
                        match_ball_velocity=True
                    )
                    
                    if self.intercept_trajectory is not None:
                        self.trajectory_start_time = sim_time
                        self.mode = 'intercept'
                        self.status = "planning"
                        des_pos = pos_np.copy()  # Start from current
                    else:
                        # Fallback to simple position control
                        des_pos = intercept_point
                        self.status = "tracking"
                else:
                    # No intercept point - stay at current
                    des_pos = pos_np.copy()
                    self.status = "waiting"
            
            # --- Step 3: Check if ball is caught ---
            if self._check_catch(pos_np) and not self.is_caught:
                self.is_caught = True
                self.status = "caught"
                print(f"[{sim_time:.3f}s] Ball CAUGHT!")
            
            # --- Step 4: Apply workspace limits ---
            des_pos_clamped, is_outside = self._clamp_to_workspace(des_pos)
            self.target_outside_workspace = is_outside
            
            # --- Step 5: Execute position control via IK ---
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
    
    def _check_catch(self, ee_pos: np.ndarray) -> bool:
        """
        Check if ball is caught by end-effector.
        
        The catch point is the sensor position (10cm above platform) 
        matching the ball position.
        
        Args:
            ee_pos: End-effector sensor position (already includes offset).
        
        Returns:
            True if ball is within catch radius.
        """
        ball_pos = self.ball_pos_log
        if np.all(ball_pos == 0):
            return False
        
        # Ball is in world frame, ee_pos is relative to base
        # Convert ee_pos to world frame for comparison
        ee_world = ee_pos.copy()
        ee_world[2] += 0.5  # Add base height
        
        distance = np.linalg.norm(ball_pos - ee_world)
        return distance < self.catch_threshold
    
    def get_log_data(self) -> Tuple:
        """
        Return log data for recording.
        
        Returns:
            Tuple of (ee_des_pos, ee_actual_pos, ball_pos, ball_vel, intercept_target, status, is_caught)
        """
        return (
            self.current_des_pos_log.copy(),
            self.current_actual_pos_log.copy(),
            self.ball_pos_log.copy(),
            self.ball_vel_log.copy(),
            self.intercept_target.copy() if self.intercept_target is not None else np.zeros(3),
            self.status,
            self.is_caught
        )
    
    def print_state(self) -> None:
        """Print current state for debugging."""
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        
        print(f"=== Delta Intercept Controller ===")
        print(f"Status: {self.status} | Caught: {self.is_caught}")
        print(f"EE Position: {self.current_actual_pos_log}")
        print(f"Ball Position: {self.ball_pos_log}")
        print(f"Ball Velocity: {self.ball_vel_log}")
        if self.intercept_target is not None:
            print(f"Intercept Target: {self.intercept_target}")
            print(f"Intercept Time: {self.intercept_time:.3f}s")
        print(f"=================================")
