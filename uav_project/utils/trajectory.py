"""
Trajectory generation utilities for UAV simulation.
"""

import torch
from typing import List, Tuple


def generate_circular_trajectory(
    center: List[float],  # Center coordinates [x, y, z]
    radius: float,        # Radius
    total_time: float,    # Total duration
    num_points: int = 50, # Number of points (default 50)
    start_angle: float = 0,  # Start angle (radians)
    clockwise: bool = False, # Direction
    height_variation: bool = False,  # Whether Z varies with angle
    height_amplitude: float = 0.5,   # Amplitude of Z variation
    hold_last: bool = True           # Whether to hold position at the end
) -> List[Tuple[float, List[float]]]:
    """
    Generates a circular trajectory.
    
    Args:
        center: Center of the circle [x, y, z].
        radius: Radius of the circle.
        total_time: Total time for the trajectory.
        num_points: Number of waypoints.
        start_angle: Starting angle in radians.
        clockwise: True for clockwise, False for counter-clockwise.
        height_variation: If True, Z height varies sinusoidally.
        height_amplitude: Amplitude of the height variation.
        hold_last: If True, adds an extra point at the end to hold position.
    
    Returns:
        A list of tuples (time, [x, y, z]).
    """
    
    trajectory = []
    direction = -1 if clockwise else 1  # -1 for clockwise, 1 for counter-clockwise
    
    # Calculate time step
    time_step = total_time / (num_points - 1) if num_points > 1 else total_time
    
    # Generate points
    for i in range(num_points):
        # Current time
        t = i * time_step
        
        # Current angle
        if num_points > 1:
            angle = torch.tensor(start_angle + direction * 2 * torch.pi * i / (num_points - 1), dtype=torch.float32)
        else:
            angle = torch.tensor(start_angle, dtype=torch.float32)
        
        # Calculate Position
        x = center[0] + radius * torch.cos(angle)
        y = center[1] + radius * torch.sin(angle)
        
        # Calculate Z
        if height_variation:
            z = center[2] + height_amplitude * torch.sin(angle * 2) 
        else:
            z = torch.tensor(center[2], dtype=torch.float32)
        
        trajectory.append((t, [x.item(), y.item(), z.item()]))
    
    # Hold last position
    if hold_last and trajectory:
        last_point = trajectory[-1]
        hold_time = last_point[0] + 1.0  # Hold for 1 second
        trajectory.append((hold_time, last_point[1]))
    
    return trajectory


def generate_spiral_trajectory(
    center: List[float],    # Start center point
    start_radius: float,    # Start radius
    end_radius: float,      # End radius
    total_time: float,      # Total time
    num_turns: int = 2,     # Number of spiral turns
    num_points: int = 100,  # Total points
    clockwise: bool = False # Direction
) -> List[Tuple[float, List[float]]]:
    """
    Generates a spiral trajectory with varying radius.

    Args:
        center: Center position [x, y, z].
        start_radius: Initial radius.
        end_radius: Final radius.
        total_time: Total duration.
        num_turns: Number of full rotations.
        num_points: Number of waypoints.
        clockwise: Direction of rotation.

    Returns:
        A list of tuples (time, [x, y, z]).
    """
    trajectory = []
    direction = -1 if clockwise else 1
    
    for i in range(num_points):
        t = i * total_time / (num_points - 1) if num_points > 1 else 0
        
        # Current angle (multiple turns)
        angle = torch.tensor(direction * 2 * torch.pi * num_turns * i / (num_points - 1), dtype=torch.float32)
        
        # Current radius (linear interpolation)
        current_radius = start_radius + (end_radius - start_radius) * i / (num_points - 1)
        
        # Calculate position
        x = center[0] + current_radius * torch.cos(angle)
        y = center[1] + current_radius * torch.sin(angle)
        z = torch.tensor(center[2], dtype=torch.float32)
        
        trajectory.append((t, [x.item(), y.item(), z.item()]))
    
    return trajectory
