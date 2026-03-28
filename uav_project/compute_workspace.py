"""
Compute Delta robot workspace boundary and create smooth surface visualization.
Uses forward kinematics to find end-effector positions and fits smooth surfaces.

The computed workspace bounds are used for:
1. Ball trajectory generation (ensure balls pass through workspace)
2. Arm trajectory planning (interception point validation)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from matplotlib.colors import Normalize

# Path setup
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from uav_project.utils.DeltaKinematics import DeltaKinematics


def compute_workspace_points(kinematics: DeltaKinematics, 
                              theta_min: float = -90.0, 
                              theta_max: float = 30.0,
                              resolution: int = 60) -> np.ndarray:
    """
    Enumerate all valid joint angle combinations to find workspace points.
    
    Args:
        kinematics: DeltaKinematics instance
        theta_min: Minimum joint angle in degrees
        theta_max: Maximum joint angle in degrees
        resolution: Number of samples per joint
    
    Returns:
        Array of valid end-effector positions (N, 3)
    """
    theta_range = np.linspace(theta_min, theta_max, resolution)
    valid_points = []
    
    print(f"Computing workspace with {resolution}^3 = {resolution**3} combinations...")
    
    for i, t1 in enumerate(theta_range):
        if i % 10 == 0:
            print(f"  Progress: {i}/{resolution}")
        for t2 in theta_range:
            for t3 in theta_range:
                theta = np.array([t1, t2, t3])
                pos = kinematics.fk(theta)
                
                if isinstance(pos, int) and pos == -1:
                    continue
                valid_points.append(pos.numpy())
    
    valid_points = np.array(valid_points)
    print(f"Found {len(valid_points)} valid points")
    
    return valid_points


def compute_theoretical_bounds(points: np.ndarray, safety_margin: float = 0.005) -> dict:
    """
    Compute theoretical maximum workspace bounds with safety margin.
    
    Args:
        points: Valid workspace points (N, 3)
        safety_margin: Safety margin in meters (default 5mm)
    
    Returns:
        Dictionary with bounds information
    """
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    # Apply safety margin (shrink workspace slightly for reliability)
    bounds = {
        'x': (x_min + safety_margin, x_max - safety_margin),
        'y': (y_min + safety_margin, y_max - safety_margin),
        'z': (z_min + safety_margin, z_max - safety_margin)
    }
    
    # Compute effective workspace radius (circular approximation for XY)
    xy_radius = min(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    effective_radius = xy_radius - safety_margin
    
    print("\n" + "=" * 60)
    print("THEORETICAL MAXIMUM WORKSPACE BOUNDS")
    print("=" * 60)
    print(f"X range: [{bounds['x'][0]:.4f}, {bounds['x'][1]:.4f}] m")
    print(f"Y range: [{bounds['y'][0]:.4f}, {bounds['y'][1]:.4f}] m")
    print(f"Z range: [{bounds['z'][0]:.4f}, {bounds['z'][1]:.4f}] m")
    print(f"Effective XY radius: {effective_radius:.4f} m")
    print(f"Safety margin: {safety_margin*1000:.1f} mm")
    print("=" * 60)
    
    return {
        'bounds': bounds,
        'effective_radius': effective_radius,
        'safety_margin': safety_margin,
        'raw_points': points
    }


def create_boundary_surface(points: np.ndarray, resolution: int = 100) -> dict:
    """
    Create smooth boundary surfaces from workspace points.
    
    Uses interpolation to create continuous surfaces representing:
    - Top boundary (maximum Z at each XY position)
    - Bottom boundary (minimum Z at each XY position)
    
    Args:
        points: Valid workspace points (N, 3)
        resolution: Grid resolution for surface interpolation
    
    Returns:
        Dictionary with surface data
    """
    # Create grid for XY plane
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Group points by XY and find min/max Z
    grid_size = resolution
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)
    
    z_top = np.full((grid_size, grid_size), np.nan)
    z_bottom = np.full((grid_size, grid_size), np.nan)
    
    for i in range(grid_size):
        for j in range(grid_size):
            mask = ((points[:, 0] >= x_bins[i]) & (points[:, 0] < x_bins[i+1]) &
                    (points[:, 1] >= y_bins[j]) & (points[:, 1] < y_bins[j+1]))
            
            bin_points = points[mask]
            
            if len(bin_points) > 0:
                z_top[i, j] = bin_points[:, 2].max()
                z_bottom[i, j] = bin_points[:, 2].min()
    
    # Create coordinate grids for surface plotting
    X = (x_bins[:-1] + x_bins[1:]) / 2
    Y = (y_bins[:-1] + y_bins[1:]) / 2
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    # Interpolate missing values (NaN)
    # Create mask for valid points
    valid_top = ~np.isnan(z_top)
    valid_bottom = ~np.isnan(z_bottom)
    
    # Use griddata for interpolation
    points_valid_top = np.column_stack([X_grid[valid_top], Y_grid[valid_top]])
    values_top = z_top[valid_top]
    
    points_valid_bottom = np.column_stack([X_grid[valid_bottom], Y_grid[valid_bottom]])
    values_bottom = z_bottom[valid_bottom]
    
    # Interpolate to fill missing values
    if len(points_valid_top) > 3:
        z_top_interp = griddata(points_valid_top, values_top, (Xi, Yi), method='linear')
        z_top_smooth = gaussian_filter(np.nan_to_num(z_top_interp, nan=np.nanmean(z_top_interp)), sigma=1)
    else:
        z_top_smooth = np.full_like(Xi, np.nanmean(z_top))
    
    if len(points_valid_bottom) > 3:
        z_bottom_interp = griddata(points_valid_bottom, values_bottom, (Xi, Yi), method='linear')
        z_bottom_smooth = gaussian_filter(np.nan_to_num(z_bottom_interp, nan=np.nanmean(z_bottom_interp)), sigma=1)
    else:
        z_bottom_smooth = np.full_like(Xi, np.nanmean(z_bottom))
    
    return {
        'Xi': Xi,
        'Yi': Yi,
        'z_top': z_top_smooth,
        'z_bottom': z_bottom_smooth,
        'X_grid': X_grid,
        'Y_grid': Y_grid,
        'z_top_raw': z_top,
        'z_bottom_raw': z_bottom
    }


def plot_workspace_surfaces(workspace_data: dict, surface_data: dict, save_path: str = None):
    """
    Plot smooth surface visualization of workspace boundary.
    """
    fig = plt.figure(figsize=(18, 12))
    
    points = workspace_data['raw_points']
    
    # ========== Plot 1: 3D Surface View ==========
    ax1 = fig.add_subplot(221, projection='3d')
    
    Xi = surface_data['Xi']
    Yi = surface_data['Yi']
    z_top = surface_data['z_top']
    z_bottom = surface_data['z_bottom']
    
    # Plot top surface
    norm_top = Normalize(vmin=z_top[~np.isnan(z_top)].min(), 
                         vmax=z_top[~np.isnan(z_top)].max())
    surf1 = ax1.plot_surface(Xi, Yi, z_top, cmap='coolwarm', alpha=0.8,
                              linewidth=0, antialiased=True, 
                              facecolors=cm.coolwarm(norm_top(z_top)))
    
    # Plot bottom surface
    surf2 = ax1.plot_surface(Xi, Yi, z_bottom, cmap='viridis', alpha=0.6,
                              linewidth=0, antialiased=True)
    
    # Plot sample points (thin scatter)
    if len(points) > 5000:
        idx = np.random.choice(len(points), 5000, replace=False)
        sample_points = points[idx]
    else:
        sample_points = points
    
    ax1.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
               c='gray', s=0.1, alpha=0.1, label='Valid points')
    
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.set_title('Workspace Boundary Surfaces\n(Top: red-white, Bottom: green-blue)', fontsize=12)
    
    # Add colorbars
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Z top (m)')
    
    # ========== Plot 2: Top View (XY) ==========
    ax2 = fig.add_subplot(222)
    
    # Create filled contour for XY workspace extent
    z_valid = ~np.isnan(z_top)
    ax2.contourf(Xi, Yi, z_valid.astype(float), levels=[0.5, 1.5], 
                 colors=['lightblue'], alpha=0.5)
    ax2.contour(Xi, Yi, z_valid.astype(float), levels=[0.5], 
                colors=['blue'], linewidths=2)
    
    # Scatter points colored by Z
    scatter = ax2.scatter(points[:, 0], points[:, 1], c=points[:, 2], 
                         cmap='viridis', s=0.5, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Z (m)')
    
    # Draw effective workspace circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    r = workspace_data['effective_radius']
    ax2.plot(r * np.cos(theta_circle), r * np.sin(theta_circle), 
            'r--', linewidth=2, label=f'Effective radius: {r*100:.1f}cm')
    
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_title('Top View (XY Plane)', fontsize=12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # ========== Plot 3: Side View (XZ) ==========
    ax3 = fig.add_subplot(223)
    
    # Find points near Y=0 for XZ cross-section
    y_tolerance = (points[:, 1].max() - points[:, 1].min()) / 30
    mask_y = np.abs(points[:, 1]) < y_tolerance
    slice_points = points[mask_y]
    
    if len(slice_points) > 0:
        # Create boundary lines
        ax3.scatter(slice_points[:, 0], slice_points[:, 2], c='blue', s=0.5, alpha=0.3)
        
        # Sort and plot boundary
        sorted_idx = np.argsort(slice_points[:, 0])
        
        # Top boundary
        x_unique = np.unique(slice_points[:, 0])
        z_top_slice = []
        z_bottom_slice = []
        for x in x_unique:
            mask_x = np.abs(slice_points[:, 0] - x) < 0.005
            if np.any(mask_x):
                z_top_slice.append(slice_points[mask_x, 2].max())
                z_bottom_slice.append(slice_points[mask_x, 2].min())
            else:
                z_top_slice.append(np.nan)
                z_bottom_slice.append(np.nan)
        
        z_top_slice = np.array(z_top_slice)
        z_bottom_slice = np.array(z_bottom_slice)
        
        valid = ~np.isnan(z_top_slice)
        if np.sum(valid) > 2:
            ax3.fill_between(x_unique[valid], z_bottom_slice[valid], z_top_slice[valid],
                           alpha=0.3, color='lightblue', label='Workspace')
            ax3.plot(x_unique[valid], z_top_slice[valid], 'b-', linewidth=1.5, label='Top boundary')
            ax3.plot(x_unique[valid], z_bottom_slice[valid], 'b-', linewidth=1.5, label='Bottom boundary')
    
    ax3.axhline(y=workspace_data['bounds']['z'][0], color='g', linestyle='--', 
               label=f"Z min: {workspace_data['bounds']['z'][0]*100:.1f}cm")
    ax3.axhline(y=workspace_data['bounds']['z'][1], color='r', linestyle='--',
               label=f"Z max: {workspace_data['bounds']['z'][1]*100:.1f}cm")
    
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Z (m)', fontsize=10)
    ax3.set_title('Side View (XZ Cross-section at Y=0)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)
    
    # ========== Plot 4: Workspace Statistics ==========
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    bounds = workspace_data['bounds']
    stats_text = f"""
WORKSPACE SPECIFICATIONS
{'='*40}

THEORETICAL BOUNDS (with safety margin):
  X: [{bounds['x'][0]:.4f}, {bounds['x'][1]:.4f}] m
  Y: [{bounds['y'][0]:.4f}, {bounds['y'][1]:.4f}] m  
  Z: [{bounds['z'][0]:.4f}, {bounds['z'][1]:.4f}] m

EFFECTIVE XY RADIUS: {workspace_data['effective_radius']*100:.2f} cm
SAFETY MARGIN: {workspace_data['safety_margin']*1000:.1f} mm

RECOMMENDED FOR TRAJECTORY PLANNING:
  - Intercept zone Z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}] m
  - Max XY distance: {workspace_data['effective_radius']:.4f} m
  
USAGE IN CODE:
  workspace_bounds = {{
      'x': {bounds['x']},
      'y': {bounds['y']},
      'z': {bounds['z']}
  }}
"""
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved workspace surface plot to: {save_path}")
    
    plt.close()
    
    return fig


def save_workspace_config(workspace_data: dict, save_dir: str):
    """
    Save workspace configuration to Python config file for use in other modules.
    """
    bounds = workspace_data['bounds']
    effective_radius = workspace_data['effective_radius']
    
    config_content = f'''"""
Delta Robot Workspace Configuration
Auto-generated from compute_workspace.py

These bounds represent the theoretical maximum workspace
with a safety margin of {workspace_data["safety_margin"]*1000:.1f} mm.

Usage:
    from uav_project.config_workspace import WORKSPACE_BOUNDS, WORKSPACE_RADIUS
    
    # Check if point is in workspace
    if (WORKSPACE_BOUNDS['x'][0] <= x <= WORKSPACE_BOUNDS['x'][1] and
        WORKSPACE_BOUNDS['y'][0] <= y <= WORKSPACE_BOUNDS['y'][1] and
        WORKSPACE_BOUNDS['z'][0] <= z <= WORKSPACE_BOUNDS['z'][1]):
        # Point is valid
        pass
"""

# Workspace bounds (meters)
WORKSPACE_BOUNDS = {{
    'x': {bounds['x']},
    'y': {bounds['y']},
    'z': {bounds['z']}
}}

# Effective XY radius for circular approximation
WORKSPACE_RADIUS = {effective_radius}

# Safety margin used in computation
SAFETY_MARGIN = {workspace_data['safety_margin']}

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
'''
    
    config_path = os.path.join(save_dir, 'config_workspace.py')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Saved workspace config to: {config_path}")


def main():
    print("=" * 60)
    print("Delta Robot Workspace Surface Computation")
    print("=" * 60)
    
    # Initialize kinematics with robot parameters
    # Parameters from Delta_Arm.xml
    kinematics = DeltaKinematics(
        rod_b=0.1,      # Upper arm length
        rod_ee=0.2,     # Lower arm length (forearm)
        r_b=0.074577,   # Base radius
        r_ee=0.02495    # End effector radius
    )
    
    # Motor angle range from XML: ctrlrange="-1.57 0.523" rad = [-90°, 30°]
    theta_min = -90.0
    theta_max = 30.0
    
    # Compute workspace points
    points = compute_workspace_points(
        kinematics, 
        theta_min=theta_min, 
        theta_max=theta_max,
        resolution=60
    )
    
    # Compute theoretical bounds with safety margin
    workspace_data = compute_theoretical_bounds(points, safety_margin=0.005)
    
    # Create smooth boundary surfaces
    print("\nCreating boundary surfaces...")
    surface_data = create_boundary_surface(points, resolution=100)
    
    # Plot smooth surfaces
    plot_path = os.path.join(current_file_dir, 'workspace_surface.png')
    plot_workspace_surfaces(workspace_data, surface_data, save_path=plot_path)
    
    # Save workspace configuration
    save_workspace_config(workspace_data, current_file_dir)
    
    # Also save raw data for potential future use
    data_path = os.path.join(current_file_dir, 'workspace_data.npz')
    np.savez(data_path, 
             all_points=points,
             bounds=workspace_data['bounds'])
    print(f"Saved workspace data to: {data_path}")
    
    print("\n" + "=" * 60)
    print("Workspace computation complete!")
    print("=" * 60)
    
    return workspace_data, surface_data


if __name__ == "__main__":
    main()
