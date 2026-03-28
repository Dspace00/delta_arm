"""
Delta Robot Workspace Configuration
Auto-generated from compute_workspace.py

These bounds represent the theoretical maximum workspace
with a safety margin of 5.0 mm.

IMPORTANT: Workspace bounds are relative to Delta base frame.
           Use BASE_HEIGHT to convert to world coordinates.

Usage:
    from uav_project.config_workspace import WORKSPACE_BOUNDS, WORKSPACE_RADIUS, BASE_HEIGHT
    
    # Check if point is in workspace (relative to base)
    if (WORKSPACE_BOUNDS['x'][0] <= x <= WORKSPACE_BOUNDS['x'][1] and
        WORKSPACE_BOUNDS['y'][0] <= y <= WORKSPACE_BOUNDS['y'][1] and
        WORKSPACE_BOUNDS['z'][0] <= z <= WORKSPACE_BOUNDS['z'][1]):
        # Point is valid
        pass
    
    # Convert to world coordinates
    world_z = BASE_HEIGHT + local_z
"""

# ========== Base Height in World Frame ==========
# Delta base is mounted at z=4.0m (moved up 3.5m from original 0.5m)
BASE_HEIGHT = 4.0  # meters

# Workspace bounds (meters, relative to base frame)
WORKSPACE_BOUNDS = {
    'x': (-0.06993047416210174, 0.06993047416210174),
    'y': (-0.08152225881814956, 0.05450085818767548),
    'z': (-0.19142920792102813, -0.09182183921337128)
}

# Effective XY radius for circular approximation
WORKSPACE_RADIUS = 0.05450085818767548

# Safety margin used in computation
SAFETY_MARGIN = 0.005

# Convenience functions
def is_in_workspace(x: float, y: float, z: float) -> bool:
    """Check if a point is within workspace bounds."""
    return (WORKSPACE_BOUNDS['x'][0] <= x <= WORKSPACE_BOUNDS['x'][1] and
            WORKSPACE_BOUNDS['y'][0] <= y <= WORKSPACE_BOUNDS['y'][1] and
            WORKSPACE_BOUNDS['z'][0] <= z <= WORKSPACE_BOUNDS['z'][1])

def clamp_to_workspace(x: float, y: float, z: float) -> tuple:
    """Clamp a point to workspace bounds."""
    x_clamped = max(WORKSPACE_BOUNDS['x'][0], min(x, WORKSPACE_BOUNDS['x'][1]))
    y_clamped = max(WORKSPACE_BOUNDS['y'][0], min(y, WORKSPACE_BOUNDS['y'][1]))
    z_clamped = max(WORKSPACE_BOUNDS['z'][0], min(z, WORKSPACE_BOUNDS['z'][1]))
    return x_clamped, y_clamped, z_clamped
