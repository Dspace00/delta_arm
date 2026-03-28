"""
Delta Arm Controller for standalone Delta robot simulation.
Extends DeltaController with external target input and workspace limits.
"""

import torch
import numpy as np
from uav_project.controllers.delta_controller import DeltaController
from uav_project.utils.DeltaKinematics import DeltaKinematics


class DeltaArmController(DeltaController):
    """
    Controller for standalone Delta robot arm simulation.
    
    Extends DeltaController with:
    - External target position input (for trajectory/joystick control)
    - Workspace boundary limits
    - Logger interface compatibility
    
    Compatible with both UAVModel and DeltaModel.
    """
    
    def __init__(self, model, control_freq=100.0, control_mode='position'):
        """
        Args:
            model: Instance of UAVModel or DeltaModel for MuJoCo interaction.
            control_freq: Control frequency in Hz.
            control_mode: 'position' for position control, 'velocity' for Jacobian-based velocity control.
        """
        super().__init__(model, control_freq, control_mode)
        
        # Alias for clarity (self.uav is set by parent class)
        self.model = model
        # Data access (for MuJoCo model/data access in simulation loop)
        self.data = model.data
        
        # External target position (set by set_target_position or joystick)
        self.target_position = None
        self.target_velocity = None
        
        # Workspace limits (relative to base, in meters)
        # These are approximate values based on Delta kinematics
        self.workspace_max_radius = 0.12  # Maximum horizontal radius
        self.workspace_z_min = -0.25      # Minimum Z (lowest position relative to base)
        self.workspace_z_max = 0.0        # Maximum Z (highest position, at base level)
        
        # Flag to indicate if target is outside workspace
        self.target_outside_workspace = False
        
        # Previous valid position (for stopping at boundary)
        self._prev_valid_pos = None
    
    def reset(self):
        """
        Resets the controller state.
        Call this when restarting simulation.
        """
        self.last_update_time = 0.0
        self.target_position = None
        self.target_velocity = None
        self.target_outside_workspace = False
        self._prev_valid_pos = None
        self.current_des_pos_log = np.zeros(3)
        self.current_actual_pos_log = np.zeros(3)
    
    def set_target_position(self, pos):
        """
        Sets the target position for the end-effector.
        
        Args:
            pos: Target position [x, y, z] in world frame or relative to base.
                  Can be list, numpy array, or tensor.
        """
        if isinstance(pos, torch.Tensor):
            self.target_position = pos.clone().view(3, 1)
        else:
            self.target_position = torch.tensor(pos, dtype=torch.float32).view(3, 1)
    
    def set_target_velocity(self, vel):
        """
        Sets the target velocity for the end-effector (for velocity control mode).
        
        Args:
            vel: Target velocity [vx, vy, vz].
        """
        if isinstance(vel, torch.Tensor):
            self.target_velocity = vel.clone().view(3, 1)
        else:
            self.target_velocity = torch.tensor(vel, dtype=torch.float32).view(3, 1)
    
    def _clamp_to_workspace(self, pos):
        """
        Checks if position is within workspace and clamps to boundary if outside.
        
        Args:
            pos: Position tensor or array [x, y, z] relative to base.
        
        Returns:
            tuple: (clamped_position, is_outside_workspace)
        """
        if isinstance(pos, torch.Tensor):
            pos_np = pos.squeeze().cpu().numpy()
        else:
            pos_np = np.array(pos)
        
        x, y, z = pos_np
        is_outside = False
        
        # Check horizontal radius
        r = np.sqrt(x**2 + y**2)
        if r > self.workspace_max_radius:
            # Clamp to boundary
            scale = self.workspace_max_radius / r
            x = x * scale
            y = y * scale
            is_outside = True
        
        # Check Z limits
        if z > self.workspace_z_max:
            z = self.workspace_z_max
            is_outside = True
        elif z < self.workspace_z_min:
            z = self.workspace_z_min
            is_outside = True
        
        return np.array([x, y, z]), is_outside
    
    def update(self, sim_time: float) -> None:
        """
        Main update loop. Should be called every simulation step.
        
        Args:
            sim_time (float): Current simulation time.
        """
        if sim_time >= self.last_update_time + self.dt:
            
            # Get current state from sensors (NumPy)
            pos_np = self.uav.get_ee_sensor_pos()
            vel_np = self.uav.get_ee_sensor_lin_vel()
            
            # Convert to Tensor for internal calculation
            current_pos = torch.tensor(pos_np, dtype=torch.float32).view(3, 1)
            current_vel = torch.tensor(vel_np, dtype=torch.float32).view(3, 1)
            
            # Determine desired position
            if self.target_position is not None:
                # Use external target
                des_pos = self.target_position.clone()
                des_vel = self.target_velocity if self.target_velocity is not None else torch.zeros((3, 1), dtype=torch.float32)
            else:
                # Fallback to internal trajectory
                des_pos, des_vel = self.get_circular_trajectory(sim_time)
            
            # Workspace boundary check and clamp
            des_pos_clamped, is_outside = self._clamp_to_workspace(des_pos)
            self.target_outside_workspace = is_outside
            
            if is_outside:
                # Target is outside workspace - stop at boundary
                # Use the clamped position
                des_pos_tensor = torch.tensor(des_pos_clamped, dtype=torch.float32)
                des_vel = torch.zeros((3, 1), dtype=torch.float32)  # Zero velocity at boundary
            else:
                des_pos_tensor = des_pos.squeeze()
            
            # Update log variables
            self.current_des_pos_log = des_pos_clamped.copy()
            self.current_actual_pos_log = pos_np.copy()
            
            # Calculate controls
            if self.control_mode == 'position':
                # Position control via IK
                joint_angles_deg = self.kinematics.ik(des_pos_tensor)
                
                if isinstance(joint_angles_deg, int) and joint_angles_deg == -1:
                    # IK failed (should not happen after clamping, but handle gracefully)
                    print(f"Warning: IK failed for target {des_pos_clamped}")
                else:
                    joint_angles_rad = torch.deg2rad(joint_angles_deg)
                    self.uav.set_delta_motor_positions(joint_angles_rad)
                    
            elif self.control_mode == 'velocity':
                # Velocity control via Jacobian
                motor_vels = self.calculate_motor_velocities(
                    current_pos, current_vel, 
                    torch.tensor(des_pos_clamped, dtype=torch.float32).view(3, 1), 
                    des_vel
                )
                motor_vels_np = motor_vels.squeeze().cpu().numpy()
                self.uav.set_delta_motor_velocities(motor_vels_np)
            
            self.last_update_time = sim_time
    
    def get_log_data(self):
        """
        Returns data for logging.
        Compatible with Logger.log() interface.
        
        Returns:
            tuple: (delta_des_pos, delta_actual_pos) for Logger
        """
        return self.current_des_pos_log, self.current_actual_pos_log
    
    def print_state(self):
        """
        Prints current state for debugging.
        """
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        
        print(f"=== Delta Arm Controller State ===")
        print(f"Time: {self.last_update_time:.3f}s")
        print(f"Target Position: {self.current_des_pos_log}")
        print(f"Actual Position: {self.current_actual_pos_log}")
        print(f"Position Error: {self.current_des_pos_log - self.current_actual_pos_log}")
        
        if self.target_outside_workspace:
            print(f"[WARNING] Target was outside workspace, clamped to boundary")
        print(f"=================================")
    
    def is_target_reachable(self, pos):
        """
        Checks if a position is reachable within workspace.
        
        Args:
            pos: Position [x, y, z] to check.
        
        Returns:
            bool: True if position is within workspace.
        """
        _, is_outside = self._clamp_to_workspace(pos)
        return not is_outside
