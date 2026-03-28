"""
Delta robot trajectory generation utilities.
Extends trajectory.py with Delta-specific trajectories and workspace constraints.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional

# Reuse existing trajectory functions
from uav_project.utils.trajectory import (
    generate_circular_trajectory,
    generate_spiral_trajectory
)


# =============================================================================
# Delta Robot Workspace Parameters
# =============================================================================

# Default workspace limits for Delta robot (relative to base)
DELTA_WORKSPACE_MAX_RADIUS = 0.12   # Maximum horizontal radius (m)
DELTA_WORKSPACE_Z_MIN = -0.25       # Minimum Z (m) - lowest position
DELTA_WORKSPACE_Z_MAX = 0.0         # Maximum Z (m) - at base level

# Default Delta base position in world frame
DELTA_BASE_POSITION = [0.0, 0.0, 0.5]  # Base at z=0.5m


# =============================================================================
# Workspace Utility Functions
# =============================================================================

def clamp_to_workspace(
    pos: np.ndarray,
    max_radius: float = DELTA_WORKSPACE_MAX_RADIUS,
    z_min: float = DELTA_WORKSPACE_Z_MIN,
    z_max: float = DELTA_WORKSPACE_Z_MAX
) -> Tuple[np.ndarray, bool]:
    """
    Clamps a position to be within Delta workspace.
    
    Args:
        pos: Position [x, y, z] relative to base.
        max_radius: Maximum horizontal radius.
        z_min: Minimum Z value.
        z_max: Maximum Z value.
    
    Returns:
        tuple: (clamped_position, was_outside)
            - clamped_position: Position clamped to workspace boundary.
            - was_outside: True if original position was outside workspace.
    """
    pos = np.array(pos, dtype=np.float64)
    x, y, z = pos
    was_outside = False
    
    # Check horizontal radius
    r = np.sqrt(x**2 + y**2)
    if r > max_radius:
        scale = max_radius / r
        x = x * scale
        y = y * scale
        was_outside = True
    
    # Check Z limits
    if z > z_max:
        z = z_max
        was_outside = True
    elif z < z_min:
        z = z_min
        was_outside = True
    
    return np.array([x, y, z]), was_outside


def is_in_workspace(
    pos: np.ndarray,
    max_radius: float = DELTA_WORKSPACE_MAX_RADIUS,
    z_min: float = DELTA_WORKSPACE_Z_MIN,
    z_max: float = DELTA_WORKSPACE_Z_MAX
) -> bool:
    """
    Checks if a position is within Delta workspace.
    
    Args:
        pos: Position [x, y, z] relative to base.
        max_radius: Maximum horizontal radius.
        z_min: Minimum Z value.
        z_max: Maximum Z value.
    
    Returns:
        bool: True if position is within workspace.
    """
    _, was_outside = clamp_to_workspace(pos, max_radius, z_min, z_max)
    return not was_outside


# =============================================================================
# Coordinate Conversion Functions
# =============================================================================

def world_to_delta_frame(
    world_pos: List[float],
    base_pos: List[float] = DELTA_BASE_POSITION
) -> np.ndarray:
    """
    Converts world frame position to Delta frame (relative to base).
    
    Args:
        world_pos: Position [x, y, z] in world frame.
        base_pos: Base position in world frame.
    
    Returns:
        np.ndarray: Position relative to base [x, y, z].
    """
    return np.array(world_pos) - np.array(base_pos)


def delta_to_world_frame(
    delta_pos: List[float],
    base_pos: List[float] = DELTA_BASE_POSITION
) -> np.ndarray:
    """
    Converts Delta frame position to world frame.
    
    Args:
        delta_pos: Position [x, y, z] relative to base.
        base_pos: Base position in world frame.
    
    Returns:
        np.ndarray: Position in world frame [x, y, z].
    """
    return np.array(delta_pos) + np.array(base_pos)


def convert_trajectory_to_delta_frame(
    trajectory: List[Tuple[float, List[float]]],
    base_pos: List[float] = DELTA_BASE_POSITION
) -> List[Tuple[float, List[float]]]:
    """
    Converts a trajectory from world frame to Delta frame.
    
    Args:
        trajectory: List of (time, [x, y, z]) in world frame.
        base_pos: Base position in world frame.
    
    Returns:
        List of (time, [x, y, z]) relative to base.
    """
    return [(t, world_to_delta_frame(pos, base_pos).tolist()) for t, pos in trajectory]


# =============================================================================
# Delta-Specific Trajectory Generators
# =============================================================================

def generate_delta_circular_trajectory(
    center_offset: List[float] = [0.0, 0.0, -0.18],
    radius: float = 0.08,
    total_time: float = 10.0,
    num_points: int = 100,
    clockwise: bool = False,
    validate_workspace: bool = True
) -> List[Tuple[float, List[float]]]:
    """
    Generates a circular trajectory for Delta robot within workspace.
    
    Args:
        center_offset: Center [x, y, z] relative to base.
        radius: Circle radius (will be clamped if too large).
        total_time: Total trajectory duration.
        num_points: Number of waypoints.
        clockwise: Rotation direction.
        validate_workspace: If True, validates all points are in workspace.
    
    Returns:
        List of (time, [x, y, z]) tuples relative to base.
    """
    # Clamp radius to workspace
    max_valid_radius = DELTA_WORKSPACE_MAX_RADIUS - np.sqrt(center_offset[0]**2 + center_offset[1]**2)
    radius = min(radius, max(0.01, max_valid_radius))
    
    # Center in Delta frame (relative to base)
    center = center_offset
    
    trajectory = []
    direction = -1 if clockwise else 1
    time_step = total_time / (num_points - 1) if num_points > 1 else total_time
    
    for i in range(num_points):
        t = i * time_step
        angle = direction * 2 * torch.pi * i / (num_points - 1) if num_points > 1 else 0
        
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2]
        
        # Validate workspace
        if validate_workspace:
            pos, outside = clamp_to_workspace([x, y, z])
            if outside:
                x, y, z = pos
        
        trajectory.append((t, [x, y, z]))
    
    return trajectory


def generate_linear_trajectory(
    start: List[float],
    end: List[float],
    total_time: float = 5.0,
    num_points: int = 50,
    validate_workspace: bool = True
) -> List[Tuple[float, List[float]]]:
    """
    Generates a linear (straight line) trajectory.
    
    Args:
        start: Start position [x, y, z] relative to base.
        end: End position [x, y, z] relative to base.
        total_time: Total trajectory duration.
        num_points: Number of waypoints.
        validate_workspace: If True, clamps points to workspace.
    
    Returns:
        List of (time, [x, y, z]) tuples relative to base.
    """
    start = np.array(start)
    end = np.array(end)
    
    trajectory = []
    time_step = total_time / (num_points - 1) if num_points > 1 else total_time
    
    for i in range(num_points):
        t = i * time_step
        alpha = i / (num_points - 1) if num_points > 1 else 0
        
        pos = start + alpha * (end - start)
        
        if validate_workspace:
            pos, _ = clamp_to_workspace(pos)
        
        trajectory.append((t, pos.tolist()))
    
    return trajectory


def generate_square_trajectory(
    center: List[float] = [0.0, 0.0, -0.18],
    side_length: float = 0.08,
    total_time: float = 12.0,
    num_points_per_side: int = 25,
    validate_workspace: bool = True
) -> List[Tuple[float, List[float]]]:
    """
    Generates a square trajectory for Delta robot.
    
    Args:
        center: Center [x, y, z] relative to base.
        side_length: Length of each side.
        total_time: Total trajectory duration.
        num_points_per_side: Points per side.
        validate_workspace: If True, clamps points to workspace.
    
    Returns:
        List of (time, [x, y, z]) tuples relative to base.
    """
    center = np.array(center)
    half_side = side_length / 2
    
    # Define corners
    corners = [
        center + np.array([-half_side, -half_side, 0]),  # Bottom-left
        center + np.array([half_side, -half_side, 0]),   # Bottom-right
        center + np.array([half_side, half_side, 0]),    # Top-right
        center + np.array([-half_side, half_side, 0]),   # Top-left
        center + np.array([-half_side, -half_side, 0]),  # Back to start
    ]
    
    trajectory = []
    time_per_side = total_time / 4
    time_step = time_per_side / (num_points_per_side - 1)
    
    t = 0
    for i in range(4):
        start_corner = corners[i]
        end_corner = corners[i + 1]
        
        for j in range(num_points_per_side):
            alpha = j / (num_points_per_side - 1)
            pos = start_corner + alpha * (end_corner - start_corner)
            
            if validate_workspace:
                pos, _ = clamp_to_workspace(pos)
            
            trajectory.append((t, pos.tolist()))
            t += time_step
    
    return trajectory


def generate_point_to_point_trajectory(
    start: List[float],
    end: List[float],
    total_time: float = 3.0,
    num_points: int = 50,
    trajectory_type: str = 'linear',
    validate_workspace: bool = True
) -> List[Tuple[float, List[float]]]:
    """
    Generates point-to-point trajectory with smooth acceleration.
    
    Args:
        start: Start position [x, y, z] relative to base.
        end: End position [x, y, z] relative to base.
        total_time: Total trajectory duration.
        num_points: Number of waypoints.
        trajectory_type: 'linear' or 'smooth' (S-curve velocity).
        validate_workspace: If True, clamps points to workspace.
    
    Returns:
        List of (time, [x, y, z]) tuples relative to base.
    """
    start = np.array(start)
    end = np.array(end)
    
    trajectory = []
    time_step = total_time / (num_points - 1) if num_points > 1 else total_time
    
    for i in range(num_points):
        t = i * time_step
        s = i / (num_points - 1) if num_points > 1 else 0
        
        if trajectory_type == 'smooth':
            # S-curve velocity profile (smooth acceleration/deceleration)
            # Using cosine interpolation: s = (1 - cos(pi * s)) / 2
            s = (1 - np.cos(np.pi * s)) / 2
        
        pos = start + s * (end - start)
        
        if validate_workspace:
            pos, _ = clamp_to_workspace(pos)
        
        trajectory.append((t, pos.tolist()))
    
    return trajectory


def generate_stay_trajectory(
    position: List[float],
    duration: float = 5.0,
    num_points: int = 10
) -> List[Tuple[float, List[float]]]:
    """
    Generates a trajectory that stays at a single position.
    
    Args:
        position: Position [x, y, z] relative to base.
        duration: Duration to stay.
        num_points: Number of points (for logging purposes).
    
    Returns:
        List of (time, [x, y, z]) tuples.
    """
    position, _ = clamp_to_workspace(position)
    
    trajectory = []
    time_step = duration / (num_points - 1) if num_points > 1 else duration
    
    for i in range(num_points):
        t = i * time_step
        trajectory.append((t, position.tolist()))
    
    return trajectory


# =============================================================================
# Trajectory Validation Utilities
# =============================================================================

def validate_trajectory(
    trajectory: List[Tuple[float, List[float]]],
    verbose: bool = False
) -> Tuple[bool, List[int]]:
    """
    Validates that all points in a trajectory are within Delta workspace.
    
    Args:
        trajectory: List of (time, [x, y, z]) tuples.
        verbose: If True, prints details about invalid points.
    
    Returns:
        tuple: (is_valid, invalid_indices)
            - is_valid: True if all points are in workspace.
            - invalid_indices: List of indices of invalid points.
    """
    invalid_indices = []
    
    for i, (t, pos) in enumerate(trajectory):
        if not is_in_workspace(pos):
            invalid_indices.append(i)
            if verbose:
                print(f"Point {i} at t={t:.2f}s is outside workspace: {pos}")
    
    return len(invalid_indices) == 0, invalid_indices


def clamp_trajectory(
    trajectory: List[Tuple[float, List[float]]]
) -> List[Tuple[float, List[float]]]:
    """
    Clamps all points in a trajectory to Delta workspace.
    
    Args:
        trajectory: List of (time, [x, y, z]) tuples.
    
    Returns:
        Clamped trajectory.
    """
    return [(t, clamp_to_workspace(pos)[0].tolist()) for t, pos in trajectory]
