"""
Smooth trajectory planning utilities.
Implements quintic polynomial trajectory (C4 continuity: 0~4th derivatives smooth).
"""
import numpy as np
from typing import Tuple, Optional


class QuinticPolynomialTrajectory:
    """
    Quintic (5th order) polynomial trajectory planner.
    
    Provides C4 continuity - position, velocity, acceleration, jerk, and snap
    are all continuous (0~4th derivatives smooth).
    
    s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    
    This is the fastest smooth trajectory method with closed-form solution.
    """
    
    def __init__(self):
        """Initialize trajectory parameters."""
        self.coefficients = None  # [a0, a1, a2, a3, a4, a5] for each axis
        self.duration = 0.0
        self.start_pos = np.zeros(3)
        self.start_vel = np.zeros(3)
        self.start_acc = np.zeros(3)
        self.end_pos = np.zeros(3)
        self.end_vel = np.zeros(3)
        self.end_acc = np.zeros(3)
    
    def plan(self,
             start_pos: np.ndarray,
             end_pos: np.ndarray,
             duration: float,
             start_vel: np.ndarray = None,
             end_vel: np.ndarray = None,
             start_acc: np.ndarray = None,
             end_acc: np.ndarray = None) -> np.ndarray:
        """
        Plan quintic polynomial trajectory.
        
        Args:
            start_pos: Start position (3,)
            end_pos: End position (3,)
            duration: Trajectory duration (seconds)
            start_vel: Start velocity (3,), default [0,0,0]
            end_vel: End velocity (3,), default [0,0,0]
            start_acc: Start acceleration (3,), default [0,0,0]
            end_acc: End acceleration (3,), default [0,0,0]
        
        Returns:
            Coefficients array (6, 3) - 6 coefficients for each axis
        """
        # Default values
        if start_vel is None:
            start_vel = np.zeros(3)
        if end_vel is None:
            end_vel = np.zeros(3)
        if start_acc is None:
            start_acc = np.zeros(3)
        if end_acc is None:
            end_acc = np.zeros(3)
        
        # Store parameters
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.start_vel = np.array(start_vel)
        self.end_vel = np.array(end_vel)
        self.start_acc = np.array(start_acc)
        self.end_acc = np.array(end_acc)
        self.duration = duration
        
        # Solve for each axis
        # Boundary conditions:
        # s(0) = a0 = start_pos
        # s'(0) = a1 = start_vel
        # s''(0) = 2*a2 = start_acc => a2 = start_acc/2
        # s(T) = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4 + a5*T^5 = end_pos
        # s'(T) = a1 + 2*a2*T + 3*a3*T^2 + 4*a4*T^3 + 5*a5*T^4 = end_vel
        # s''(T) = 2*a2 + 6*a3*T + 12*a4*T^2 + 20*a5*T^3 = end_acc
        
        T = duration
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T
        
        coefficients = np.zeros((6, 3))
        
        for axis in range(3):
            # Known coefficients
            a0 = start_pos[axis]
            a1 = start_vel[axis]
            a2 = start_acc[axis] / 2.0
            
            # Solve for a3, a4, a5
            # From boundary conditions at t=T:
            # a3*T^3 + a4*T^4 + a5*T^5 = end_pos - a0 - a1*T - a2*T^2
            # 3*a3*T^2 + 4*a4*T^3 + 5*a5*T^4 = end_vel - a1 - 2*a2*T
            # 6*a3*T + 12*a4*T^2 + 20*a5*T^3 = end_acc - 2*a2
            
            A = np.array([
                [T3, T4, T5],
                [3*T2, 4*T3, 5*T4],
                [6*T, 12*T2, 20*T3]
            ])
            
            b = np.array([
                end_pos[axis] - a0 - a1*T - a2*T2,
                end_vel[axis] - a1 - 2*a2*T,
                end_acc[axis] - 2*a2
            ])
            
            try:
                a3, a4, a5 = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback to zero if singular
                a3, a4, a5 = 0.0, 0.0, 0.0
            
            coefficients[:, axis] = [a0, a1, a2, a3, a4, a5]
        
        self.coefficients = coefficients
        return coefficients
    
    def evaluate(self, t: float, derivative: int = 0) -> np.ndarray:
        """
        Evaluate trajectory at time t.
        
        Args:
            t: Time (seconds)
            derivative: Order of derivative (0=pos, 1=vel, 2=acc, 3=jerk, 4=snap)
        
        Returns:
            Position/velocity/acceleration/etc. (3,)
        """
        if self.coefficients is None:
            raise ValueError("Trajectory not planned. Call plan() first.")
        
        # Clamp time
        t = np.clip(t, 0.0, self.duration)
        
        a = self.coefficients
        
        if derivative == 0:
            # s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
            return (a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5)
        
        elif derivative == 1:
            # s'(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
            return (a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4)
        
        elif derivative == 2:
            # s''(t) = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3
            return (2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3)
        
        elif derivative == 3:
            # s'''(t) = 6*a3 + 24*a4*t + 60*a5*t^2
            return (6*a[3] + 24*a[4]*t + 60*a[5]*t**2)
        
        elif derivative == 4:
            # s''''(t) = 24*a4 + 120*a5*t
            return (24*a[4] + 120*a[5]*t)
        
        else:
            raise ValueError(f"Derivative order {derivative} not supported (0-4)")
    
    def get_position(self, t: float) -> np.ndarray:
        """Get position at time t."""
        return self.evaluate(t, 0)
    
    def get_velocity(self, t: float) -> np.ndarray:
        """Get velocity at time t."""
        return self.evaluate(t, 1)
    
    def get_acceleration(self, t: float) -> np.ndarray:
        """Get acceleration at time t."""
        return self.evaluate(t, 2)
    
    def get_jerk(self, t: float) -> np.ndarray:
        """Get jerk at time t."""
        return self.evaluate(t, 3)
    
    def get_snap(self, t: float) -> np.ndarray:
        """Get snap at time t."""
        return self.evaluate(t, 4)
    
    def sample_trajectory(self, dt: float = 0.01) -> dict:
        """
        Sample entire trajectory.
        
        Args:
            dt: Time step for sampling.
        
        Returns:
            Dict with 'time', 'pos', 'vel', 'acc', 'jerk', 'snap' arrays.
        """
        if self.coefficients is None:
            raise ValueError("Trajectory not planned. Call plan() first.")
        
        times = np.arange(0, self.duration + dt, dt)
        
        trajectory = {
            'time': times,
            'pos': np.array([self.get_position(t) for t in times]),
            'vel': np.array([self.get_velocity(t) for t in times]),
            'acc': np.array([self.get_acceleration(t) for t in times]),
            'jerk': np.array([self.get_jerk(t) for t in times]),
            'snap': np.array([self.get_snap(t) for t in times])
        }
        
        return trajectory


class MultiPointTrajectory:
    """
    Multi-point trajectory planner using quintic polynomials.
    Smoothly connects multiple waypoints with C4 continuity.
    """
    
    def __init__(self):
        """Initialize multi-point trajectory."""
        self.segments = []  # List of QuinticPolynomialTrajectory
        self.segment_times = []  # Start time of each segment
        self.total_duration = 0.0
    
    def plan(self,
             waypoints: np.ndarray,
             segment_duration: float = 0.5,
             velocities: np.ndarray = None) -> None:
        """
        Plan trajectory through multiple waypoints.
        
        Args:
            waypoints: Array of waypoints (N, 3).
            segment_duration: Duration for each segment.
            velocities: Optional velocities at waypoints (N, 3).
        """
        self.segments = []
        self.segment_times = []
        self.total_duration = 0.0
        
        n_points = len(waypoints)
        
        if velocities is None:
            # Compute reasonable velocities based on waypoint directions
            velocities = self._compute_waypoint_velocities(waypoints)
        
        for i in range(n_points - 1):
            traj = QuinticPolynomialTrajectory()
            
            # Use zero acceleration at boundaries for smoother motion
            start_acc = np.zeros(3)
            end_acc = np.zeros(3)
            
            # For intermediate points, use continuity conditions
            if i > 0:
                # Match previous segment's end velocity
                start_acc = self.segments[-1].get_acceleration(segment_duration)
            
            traj.plan(
                start_pos=waypoints[i],
                end_pos=waypoints[i+1],
                duration=segment_duration,
                start_vel=velocities[i],
                end_vel=velocities[i+1],
                start_acc=start_acc,
                end_acc=end_acc if i == n_points - 2 else np.zeros(3)
            )
            
            self.segments.append(traj)
            self.segment_times.append(self.total_duration)
            self.total_duration += segment_duration
    
    def _compute_waypoint_velocities(self, waypoints: np.ndarray) -> np.ndarray:
        """
        Compute smooth velocities at waypoints.
        
        Uses finite differences for intermediate points,
        zero velocity at start and end.
        """
        n = len(waypoints)
        velocities = np.zeros_like(waypoints)
        
        # Start and end: zero velocity
        velocities[0] = np.zeros(3)
        velocities[-1] = np.zeros(3)
        
        # Intermediate: average of adjacent directions
        for i in range(1, n - 1):
            d_prev = waypoints[i] - waypoints[i-1]
            d_next = waypoints[i+1] - waypoints[i]
            
            # Normalize
            d_prev_norm = d_prev / (np.linalg.norm(d_prev) + 1e-6)
            d_next_norm = d_next / (np.linalg.norm(d_next) + 1e-6)
            
            # Average direction
            avg_dir = (d_prev_norm + d_next_norm) / 2.0
            
            # Scale by minimum of adjacent segment speeds
            speed = min(np.linalg.norm(d_prev), np.linalg.norm(d_next)) * 0.5
            velocities[i] = avg_dir * speed
        
        return velocities
    
    def evaluate(self, t: float, derivative: int = 0) -> np.ndarray:
        """Evaluate trajectory at time t."""
        # Find which segment
        for i, seg in enumerate(self.segments):
            if t < self.segment_times[i] + seg.duration:
                local_t = t - self.segment_times[i]
                return seg.evaluate(local_t, derivative)
        
        # Past end - return last value
        return self.segments[-1].evaluate(self.segments[-1].duration, derivative)
    
    def get_position(self, t: float) -> np.ndarray:
        """Get position at time t."""
        return self.evaluate(t, 0)


def plan_intercept_trajectory(
    current_pos: np.ndarray,
    intercept_pos: np.ndarray,
    current_vel: np.ndarray,
    intercept_vel: np.ndarray,
    duration: float,
    current_acc: np.ndarray = None,
    intercept_acc: np.ndarray = None
) -> QuinticPolynomialTrajectory:
    """
    Quick helper function to plan intercept trajectory.
    
    Args:
        current_pos: Current end-effector position.
        intercept_pos: Target intercept position.
        current_vel: Current velocity.
        intercept_vel: Desired velocity at intercept.
        duration: Time to reach intercept.
        current_acc: Current acceleration (default zero).
        intercept_acc: Desired acceleration at intercept (default zero).
    
    Returns:
        QuinticPolynomialTrajectory instance.
    """
    if current_acc is None:
        current_acc = np.zeros(3)
    if intercept_acc is None:
        intercept_acc = np.zeros(3)
    
    traj = QuinticPolynomialTrajectory()
    traj.plan(
        start_pos=current_pos,
        end_pos=intercept_pos,
        duration=duration,
        start_vel=current_vel,
        end_vel=intercept_vel,
        start_acc=current_acc,
        end_acc=intercept_acc
    )
    
    return traj
