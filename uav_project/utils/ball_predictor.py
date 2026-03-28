"""
Ball trajectory predictor for Delta arm interception.
Predicts future ball position using parabolic motion model.
Integrates with smooth trajectory planning for C4 continuity.

UPDATED:
- Added 3-point trajectory fitting for improved robustness
- Added nearest-point intercept selection strategy
"""
import numpy as np
import torch
from typing import Optional, Tuple, List

from uav_project.utils.smooth_trajectory import QuinticPolynomialTrajectory, plan_intercept_trajectory


class BallPredictor:
    """
    Predicts ball trajectory using projectile motion equations.
    Assumes constant gravity and no air resistance.
    
    Provides smooth trajectory planning for interception with C4 continuity
    (0~4th derivatives are smooth).
    
    NEW: Supports 3-point trajectory fitting for improved robustness.
    """
    
    def __init__(self, gravity: float = 9.81):
        """
        Args:
            gravity: Gravitational acceleration (m/s^2)
        """
        self.gravity = gravity
        self.g = np.array([0.0, 0.0, -gravity])
        
        # Ball state (from current sensor reading)
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        
        # 3-point trajectory fitting state
        self.position_history: List[Tuple[np.ndarray, float]] = []  # [(pos, time), ...]
        self.fitted_pos0: Optional[np.ndarray] = None  # Fitted initial position
        self.fitted_vel0: Optional[np.ndarray] = None  # Fitted initial velocity
        self.fitting_error: float = 0.0  # Validation error
        
        # Prediction results
        self.predicted_trajectory: List[np.ndarray] = []
        self.intercept_point: Optional[np.ndarray] = None
        self.intercept_time: Optional[float] = None
        self.intercept_velocity: Optional[np.ndarray] = None
        
        # Smooth trajectory for interception
        self.intercept_trajectory: Optional[QuinticPolynomialTrajectory] = None
    
    def update_state(self, pos: np.ndarray, vel: np.ndarray, sim_time: float = 0.0) -> None:
        """
        Update current ball state from sensor readings.
        
        Args:
            pos: Ball position (x, y, z) in meters
            vel: Ball velocity (vx, vy, vz) in m/s
            sim_time: Current simulation time (for 3-point fitting)
        """
        self.current_pos = pos.copy()
        self.current_vel = vel.copy()
        
        # Also add to history for 3-point fitting
        if sim_time > 0:
            self.add_position_sample(pos, sim_time)
    
    def add_position_sample(self, pos: np.ndarray, time: float) -> None:
        """
        Add a position sample for 3-point trajectory fitting.
        
        Args:
            pos: Ball position (x, y, z)
            time: Time of measurement
        """
        self.position_history.append((pos.copy(), time))
        
        # Keep only last 3 samples
        if len(self.position_history) > 3:
            self.position_history.pop(0)
    
    def fit_trajectory_3points(self) -> Tuple[bool, float]:
        """
        Fit trajectory using 3 points: 2 for fitting, 1 for validation.
        
        Physics model: r(t) = r0 + v0*t + 0.5*g*t²
        
        With 2 points, we can solve for r0 and v0:
        r(t1) = r0 + v0*t1 + 0.5*g*t1²
        r(t2) = r0 + v0*t2 + 0.5*g*t2²
        
        Subtract: r(t2) - r(t1) = v0*(t2-t1) + 0.5*g*(t2² - t1²)
        
        Third point is used for validation and smoothing.
        
        Returns:
            (success, validation_error)
        """
        if len(self.position_history) < 2:
            return False, float('inf')
        
        # Get the points
        p1, t1 = self.position_history[0]
        p2, t2 = self.position_history[1]
        
        # Time difference
        dt = t2 - t1
        if abs(dt) < 0.001:  # Avoid division by zero
            return False, float('inf')
        
        # Gravity effect during dt
        gravity_effect = np.array([0, 0, -0.5 * self.gravity * (t2**2 - t1**2)])
        
        # Solve for initial velocity (at t=0, not t=t1)
        # r(t2) - r(t1) = v0 * (t2 - t1) + 0.5 * g * (t2² - t1²)
        # v0 = (r(t2) - r(t1) - 0.5*g*(t2²-t1²)) / (t2 - t1)
        v0 = (p2 - p1 - gravity_effect) / dt
        
        # Solve for initial position (at t=0)
        # r(t1) = r0 + v0*t1 + 0.5*g*t1²
        r0 = p1 - v0 * t1 - np.array([0, 0, 0.5 * self.gravity * t1**2])
        
        self.fitted_pos0 = r0
        self.fitted_vel0 = v0
        
        # Validate with third point if available
        if len(self.position_history) >= 3:
            p3, t3 = self.position_history[2]
            predicted = self.predict_from_fitted(t3)
            self.fitting_error = np.linalg.norm(predicted - p3)
            
            # If error is too large, the trajectory might be invalid
            if self.fitting_error > 0.1:  # 10cm error threshold
                return False, self.fitting_error
        
        return True, self.fitting_error
    
    def predict_from_fitted(self, t: float) -> np.ndarray:
        """
        Predict position at time t using fitted trajectory.
        
        Args:
            t: Time from start (t=0 at first sample)
        
        Returns:
            Predicted position
        """
        if self.fitted_pos0 is None or self.fitted_vel0 is None:
            return self.current_pos.copy()
        
        pos = self.fitted_pos0 + self.fitted_vel0 * t
        pos[2] -= 0.5 * self.gravity * t**2
        
        return pos
    
    def get_fitted_state(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get fitted initial position and velocity.
        
        Returns:
            (pos0, vel0) or (None, None) if not fitted
        """
        return self.fitted_pos0, self.fitted_vel0
    
    def clear_history(self) -> None:
        """Clear position history for new ball."""
        self.position_history = []
        self.fitted_pos0 = None
        self.fitted_vel0 = None
        self.fitting_error = 0.0
    
    def predict_position(self, t: float) -> np.ndarray:
        """
        Predict ball position at time t from now.
        
        Using projectile motion equations:
        x(t) = x0 + vx0 * t
        y(t) = y0 + vy0 * t  
        z(t) = z0 + vz0 * t - 0.5 * g * t^2
        
        Args:
            t: Time from now (seconds)
        
        Returns:
            Predicted position (x, y, z)
        """
        pos = self.current_pos.copy()
        pos[0] += self.current_vel[0] * t
        pos[1] += self.current_vel[1] * t
        pos[2] += self.current_vel[2] * t - 0.5 * self.gravity * t * t
        
        return pos
    
    def predict_velocity(self, t: float) -> np.ndarray:
        """
        Predict ball velocity at time t from now.
        
        v(t) = v0 + g * t
        
        Args:
            t: Time from now (seconds)
        
        Returns:
            Predicted velocity (vx, vy, vz)
        """
        vel = self.current_vel.copy()
        vel[2] -= self.gravity * t
        return vel
    
    def predict_trajectory(self, duration: float, dt: float = 0.01) -> List[np.ndarray]:
        """
        Predict full trajectory for given duration.
        
        Args:
            duration: Prediction duration (seconds)
            dt: Time step for prediction
        
        Returns:
            List of predicted positions
        """
        trajectory = []
        t = 0.0
        while t < duration:
            pos = self.predict_position(t)
            # Stop if ball hits ground (z < 0)
            if pos[2] < 0:
                break
            trajectory.append(pos)
            t += dt
        
        self.predicted_trajectory = trajectory
        return trajectory
    
    def find_intercept_point(self, 
                             workspace_bounds: dict,
                             min_reaction_time: float = 0.1,
                             max_intercept_time: float = 2.0,
                             dt: float = 0.005) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Find the best intercept point within Delta workspace.
        
        Args:
            workspace_bounds: Dict with 'x', 'y', 'z' tuples of (min, max)
            min_reaction_time: Minimum time before intercept (for arm to react)
            max_intercept_time: Maximum prediction time
            dt: Time step for searching
        
        Returns:
            (intercept_point, intercept_time) or (None, None) if not found
        """
        t = min_reaction_time
        best_point = None
        best_time = None
        
        while t < max_intercept_time:
            pos = self.predict_position(t)
            
            # Check if position is within workspace
            if self._is_in_workspace(pos, workspace_bounds):
                # Check if z is reasonable (ball still above ground)
                if pos[2] > workspace_bounds['z'][0]:
                    best_point = pos
                    best_time = t
                    break
            
            # Stop if ball hits ground
            if pos[2] < 0:
                break
            
            t += dt
        
        self.intercept_point = best_point
        self.intercept_time = best_time
        
        if best_point is not None:
            self.intercept_velocity = self.predict_velocity(best_time)
        
        return best_point, best_time
    
    def plan_smooth_intercept_trajectory(
        self,
        ee_current_pos: np.ndarray,
        ee_current_vel: np.ndarray,
        ee_current_acc: np.ndarray = None,
        match_ball_velocity: bool = True
    ) -> Optional[QuinticPolynomialTrajectory]:
        """
        Plan a C4-smooth trajectory for interception.
        
        Uses quintic polynomial for smooth position, velocity, acceleration,
        jerk, and snap (0~4th derivatives smooth).
        
        Args:
            ee_current_pos: Current end-effector position.
            ee_current_vel: Current end-effector velocity.
            ee_current_acc: Current end-effector acceleration (default zero).
            match_ball_velocity: If True, end velocity matches ball velocity at intercept.
        
        Returns:
            QuinticPolynomialTrajectory instance, or None if no intercept point.
        """
        if self.intercept_point is None or self.intercept_time is None:
            return None
        
        if ee_current_acc is None:
            ee_current_acc = np.zeros(3)
        
        # Determine end velocity
        if match_ball_velocity and self.intercept_velocity is not None:
            end_vel = self.intercept_velocity
        else:
            end_vel = np.zeros(3)
        
        # Plan quintic polynomial trajectory
        self.intercept_trajectory = plan_intercept_trajectory(
            current_pos=ee_current_pos,
            intercept_pos=self.intercept_point,
            current_vel=ee_current_vel,
            intercept_vel=end_vel,
            duration=self.intercept_time,
            current_acc=ee_current_acc,
            intercept_acc=np.zeros(3)
        )
        
        return self.intercept_trajectory
    
    def get_trajectory_position(self, t: float) -> Optional[np.ndarray]:
        """
        Get position from planned trajectory at relative time t.
        
        Args:
            t: Time since trajectory start (seconds).
        
        Returns:
            Position array or None if no trajectory planned.
        """
        if self.intercept_trajectory is None:
            return None
        return self.intercept_trajectory.get_position(t)
    
    def get_trajectory_velocity(self, t: float) -> Optional[np.ndarray]:
        """
        Get velocity from planned trajectory at relative time t.
        """
        if self.intercept_trajectory is None:
            return None
        return self.intercept_trajectory.get_velocity(t)
    
    def _is_in_workspace(self, pos: np.ndarray, bounds: dict) -> bool:
        """Check if position is within workspace bounds."""
        x_ok = bounds['x'][0] <= pos[0] <= bounds['x'][1]
        y_ok = bounds['y'][0] <= pos[1] <= bounds['y'][1]
        z_ok = bounds['z'][0] <= pos[2] <= bounds['z'][1]
        return x_ok and y_ok and z_ok
    
    def get_log_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[float]]:
        """
        Return log data for recording.
        
        Returns:
            (current_pos, current_vel, intercept_point, intercept_time)
        """
        return (
            self.current_pos.copy(),
            self.current_vel.copy(),
            self.intercept_point.copy() if self.intercept_point is not None else None,
            self.intercept_time
        )
    
    def print_state(self) -> None:
        """Print current prediction state."""
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        print(f"Ball Position: {self.current_pos}")
        print(f"Ball Velocity: {self.current_vel}")
        if self.intercept_point is not None:
            print(f"Intercept Point: {self.intercept_point}")
            print(f"Intercept Time: {self.intercept_time:.3f}s")
            print(f"Intercept Velocity: {self.intercept_velocity}")
        else:
            print("No valid intercept point found")
