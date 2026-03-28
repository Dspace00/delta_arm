"""
Optimized ball trajectory generator for Delta arm interception simulation.
Ensures balls pass through Delta workspace with proper timing.
Uses theoretical maximum workspace bounds from config_workspace.py.

UPDATED: 
- Base height changed to 4.0m (Delta mounted higher)
- Removed flight time constraints
- Added collision avoidance with Delta arm structure
"""
import numpy as np
from typing import Tuple, Optional, List

from uav_project.config_workspace import WORKSPACE_BOUNDS, WORKSPACE_RADIUS, BASE_HEIGHT


class OptimizedBallTrajectoryGenerator:
    """
    Generates ball trajectories optimized for Delta arm interception.
    
    Key features:
    1. Trajectories pass through Delta workspace
    2. Collision avoidance with Delta arm base and links
    3. No flight time constraints (any valid trajectory allowed)
    4. Random launch position within valid range
    """
    
    def __init__(self, 
                 base_height: float = None):
        """
        Args:
            base_height: Height of Delta base above ground (default: from config).
        """
        self.base_height = BASE_HEIGHT if base_height is None else base_height
        
        # Use theoretical maximum workspace bounds from config
        # These are relative to base frame
        self.workspace_bounds = WORKSPACE_BOUNDS
        self.workspace_radius = WORKSPACE_RADIUS
        
        # Convert to world frame Z range
        self.workspace_z_min = self.base_height + WORKSPACE_BOUNDS['z'][0]
        self.workspace_z_max = self.base_height + WORKSPACE_BOUNDS['z'][1]
        
        self.g = 9.81
        
        # Collision avoidance parameters
        # Delta base: cylindrical region at z=BASE_HEIGHT, radius ~0.1m
        self.base_radius = 0.12  # Base platform radius
        self.base_height_min = self.base_height - 0.05  # Base extends slightly below
        self.base_height_max = self.base_height + 0.15  # Base + motors height
        
        # Arm link radius for collision checking
        self.arm_link_radius = 0.03  # Approximate arm thickness
    
    def _check_trajectory_collision(
        self, 
        start_pos: np.ndarray, 
        start_vel: np.ndarray, 
        flight_time: float,
        dt: float = 0.02
    ) -> bool:
        """
        Check if trajectory collides with Delta base or arm links.
        
        Args:
            start_pos: Launch position.
            start_vel: Launch velocity.
            flight_time: Total flight time.
            dt: Time step for collision checking.
        
        Returns:
            True if collision detected, False otherwise.
        """
        t = 0.0
        while t < flight_time:
            # Calculate ball position at time t
            pos = np.array([
                start_pos[0] + start_vel[0] * t,
                start_pos[1] + start_vel[1] * t,
                start_pos[2] + start_vel[2] * t - 0.5 * self.g * t**2
            ])
            
            # Check collision with base (cylinder)
            xy_dist = np.sqrt(pos[0]**2 + pos[1]**2)
            if (xy_dist < self.base_radius and 
                self.base_height_min < pos[2] < self.base_height_max):
                return True  # Collision with base
            
            # Check if ball is in the region where arms might be
            # Arms extend from base to ~0.2m below base
            if (xy_dist < self.base_radius + 0.05 and 
                pos[2] > self.base_height - 0.25 and 
                pos[2] < self.base_height):
                # Check arm collision (approximate)
                # Arms are at 0°, 120°, 240° angles from base
                ball_angle = np.arctan2(pos[1], pos[0])
                for arm_angle in [0, 2*np.pi/3, 4*np.pi/3]:
                    angle_diff = abs(ball_angle - arm_angle)
                    angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                    if angle_diff < 0.3:  # Within arm's angular width
                        # Check radial distance along arm
                        if xy_dist < self.base_radius + 0.1:
                            return True  # Collision with arm
            
            t += dt
        
        return False  # No collision
    
    def generate_optimized_params(
        self, 
        ball_id: int, 
        ee_current_pos: np.ndarray,
        difficulty: float = 0.5,
        max_attempts: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Generate ball trajectory parameters with collision avoidance.
        
        New logic:
        1. Choose random intercept point within workspace
        2. Choose random launch position (any valid point)
        3. Calculate trajectory (no flight time constraints)
        4. Verify no collision with Delta arm
        
        Args:
            ball_id: Ball identifier for reproducibility.
            ee_current_pos: Current end-effector position (world frame).
            difficulty: Difficulty level (0.0 = easy, 1.0 = hard).
            max_attempts: Maximum attempts to find collision-free trajectory.
        
        Returns:
            (start_pos, start_vel, intercept_point, flight_time)
        """
        np.random.seed(42 + ball_id)
        
        # Extract current EE position (relative to base)
        ee_x = ee_current_pos[0]
        ee_y = ee_current_pos[1]
        
        for attempt in range(max_attempts):
            # ========== Step 1: Choose intercept point in workspace ==========
            # Bias toward current EE position
            spread = 0.05 + 0.05 * difficulty  # 5-10cm spread
            
            intercept_x = np.clip(
                ee_x + np.random.uniform(-spread, spread),
                -self.workspace_radius * 0.9,
                self.workspace_radius * 0.9
            )
            intercept_y = np.clip(
                ee_y + np.random.uniform(-spread, spread),
                -self.workspace_radius * 0.9,
                self.workspace_radius * 0.9
            )
            
            # Z: within workspace height range
            intercept_z = np.random.uniform(
                self.workspace_z_min + 0.01,
                self.workspace_z_max - 0.01
            )
            
            intercept_point = np.array([intercept_x, intercept_y, intercept_z])
            
            # ========== Step 2: Choose launch position ==========
            # Random distance: 2m - 15m from Delta base
            launch_distance = np.random.uniform(2.0, 15.0)
            launch_angle = np.random.uniform(0, 2 * np.pi)
            
            launch_x = launch_distance * np.cos(launch_angle)
            launch_y = launch_distance * np.sin(launch_angle)
            
            # Launch height: above intercept point, or at reasonable height
            # Allow both upward and downward throws
            launch_z = np.random.uniform(
                max(intercept_z - 2.0, 1.0),  # Not too low
                intercept_z + 3.0  # Above intercept
            )
            
            start_pos = np.array([launch_x, launch_y, launch_z])
            
            # ========== Step 3: Calculate flight time and velocity ==========
            # Horizontal distance
            dx = intercept_x - launch_x
            dy = intercept_y - launch_y
            dz = intercept_z - launch_z
            
            # For parabolic motion:
            # z(t) = z0 + vz*t - 0.5*g*t^2
            # x(t) = x0 + vx*t
            # y(t) = y0 + vy*t
            # 
            # We need to find t such that ball reaches intercept point
            # This gives us 3 equations, 4 unknowns (vx, vy, vz, t)
            # 
            # Strategy: choose reasonable horizontal speed, derive t, then vz
            
            horizontal_dist = np.sqrt(dx**2 + dy**2)
            
            # Reasonable horizontal speed: 3-20 m/s
            horizontal_speed = np.random.uniform(5.0, 18.0)
            flight_time = horizontal_dist / horizontal_speed
            
            # Minimum flight time to allow arm to react
            if flight_time < 0.3:
                flight_time = 0.3
                horizontal_speed = horizontal_dist / flight_time
            
            # Calculate velocities
            vx0 = dx / flight_time
            vy0 = dy / flight_time
            vz0 = (dz + 0.5 * self.g * flight_time**2) / flight_time
            
            start_vel = np.array([vx0, vy0, vz0])
            
            # ========== Step 4: Check collision ==========
            if self._check_trajectory_collision(start_pos, start_vel, flight_time):
                continue  # Try again
            
            # Trajectory is valid!
            return start_pos, start_vel, intercept_point, flight_time
        
        # If all attempts fail, return a safe default trajectory
        print(f"Warning: Could not find collision-free trajectory for ball {ball_id}")
        
        # Default: simple drop from above
        start_pos = np.array([0.0, 0.0, self.base_height + 1.0])
        flight_time = np.sqrt(2 * (start_pos[2] - intercept_z) / self.g)
        start_vel = np.array([0.0, 0.0, 0.0])
        
        return start_pos, start_vel, intercept_point, flight_time
    
    def verify_trajectory(
        self, 
        start_pos: np.ndarray, 
        start_vel: np.ndarray,
        intercept_point: np.ndarray,
        flight_time: float
    ) -> Tuple[bool, float]:
        """
        Verify that trajectory passes through intercept point.
        
        Args:
            start_pos: Launch position.
            start_vel: Launch velocity.
            intercept_point: Expected intercept point.
            flight_time: Expected flight time.
        
        Returns:
            (is_valid, error_distance)
        """
        # Predict position at flight_time
        actual_pos = start_pos.copy()
        actual_pos[0] += start_vel[0] * flight_time
        actual_pos[1] += start_vel[1] * flight_time
        actual_pos[2] += start_vel[2] * flight_time - 0.5 * self.g * flight_time**2
        
        error = np.linalg.norm(actual_pos - intercept_point)
        is_valid = error < 0.01  # 1cm tolerance
        
        return is_valid, error
    
    def generate_batch(
        self, 
        n_balls: int, 
        ee_start_pos: np.ndarray,
        difficulty_progression: bool = True
    ) -> list:
        """
        Generate a batch of ball trajectories.
        
        Args:
            n_balls: Number of balls to generate.
            ee_start_pos: Starting end-effector position.
            difficulty_progression: If True, difficulty increases over time.
        
        Returns:
            List of (start_pos, start_vel, intercept_point, flight_time, ball_id)
        """
        trajectories = []
        current_ee_pos = ee_start_pos.copy()
        
        for ball_id in range(n_balls):
            # Determine difficulty
            if difficulty_progression:
                difficulty = min(1.0, ball_id / (n_balls * 0.5))
            else:
                difficulty = 0.5
            
            # Generate trajectory
            start_pos, start_vel, intercept_point, flight_time = \
                self.generate_optimized_params(ball_id, current_ee_pos, difficulty)
            
            # Verify
            is_valid, error = self.verify_trajectory(
                start_pos, start_vel, intercept_point, flight_time
            )
            
            if not is_valid:
                print(f"Warning: Ball {ball_id} trajectory error: {error:.4f}m")
            
            trajectories.append((
                start_pos, start_vel, intercept_point, flight_time, ball_id
            ))
            
            # Update EE position for next ball (assume catch at intercept point)
            current_ee_pos = intercept_point.copy()
        
        return trajectories


class TrajectoryDifficultyAnalyzer:
    """Analyzes trajectory difficulty for catch rate optimization."""
    
    @staticmethod
    def calculate_difficulty(
        intercept_point: np.ndarray,
        flight_time: float,
        ee_current_pos: np.ndarray,
        workspace_radius: float = 0.10
    ) -> float:
        """
        Calculate difficulty score for a trajectory.
        
        Higher score = more difficult to catch.
        
        Args:
            intercept_point: Ball intercept position.
            flight_time: Time until intercept.
            ee_current_pos: Current end-effector position.
            workspace_radius: Workspace radius.
        
        Returns:
            Difficulty score (0.0 = easy, 1.0 = hard).
        """
        # Distance to travel
        distance = np.linalg.norm(intercept_point - ee_current_pos)
        distance_score = min(1.0, distance / 0.15)  # Normalize to 15cm max
        
        # Time pressure
        time_score = 1.0 - min(1.0, flight_time / 0.5)  # Less time = harder
        
        # Workspace edge difficulty
        edge_dist = workspace_radius - np.sqrt(intercept_point[0]**2 + intercept_point[1]**2)
        edge_score = 1.0 - min(1.0, edge_dist / workspace_radius)
        
        # Combined score
        difficulty = 0.4 * distance_score + 0.4 * time_score + 0.2 * edge_score
        
        return np.clip(difficulty, 0.0, 1.0)
    
    @staticmethod
    def get_difficulty_category(difficulty: float) -> str:
        """Get difficulty category name."""
        if difficulty < 0.3:
            return "Easy"
        elif difficulty < 0.6:
            return "Medium"
        else:
            return "Hard"
