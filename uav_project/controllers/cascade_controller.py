"""
Cascade PID Controller implementation for Quadrotor.
Structure: Position -> Velocity -> Attitude -> Angular Rate -> Mixer.
"""

import torch
import numpy as np
import quaternion
from ..config import (
    MASS, GRAVITY,
    FREQ_POSITION, FREQ_VELOCITY, FREQ_ATTITUDE, FREQ_ANGLE_RATE,
    POS_KP, VEL_KP, VEL_KI, VEL_KD,
    ATT_KP, RATE_KP, RATE_KI, RATE_KD
)
from .pid import PosPID, VelPID, AttitudePID3D, AngVelPID
from ..models.mixer import Mixer

class CascadeController:
    """
    Manages the cascade of PID controllers to control the UAV.
    """
    def __init__(self, uav_model):
        """
        Args:
            uav_model: Instance of UAVModel to interact with the drone.
        """
        self.uav = uav_model
        self.mixer = Mixer()
        
        # Get actual UAV mass from model
        # The UAV_body is not the root of the kinematic tree for the delta arm in some XML configurations
        # (e.g. if the delta base is welded/connected rather than a direct child body).
        # To be safe and accurately reflect the total physics mass of the entire model
        # we calculate the sum of all body masses in the mujoco model.
        if hasattr(self.uav, 'model'):
            # Calculate total mass of all bodies in the kinematic tree
            # Filter out worldbody (mass 0) and any non-dynamic bodies
            self.mass = sum(m for m in self.uav.model.body_mass if m > 0)
            print(f"Detected total system mass from MuJoCo: {self.mass:.3f} kg")
        else:
            self.mass = MASS
        
        # Simulation time tracking
        self.sim_time = 0.0
        
        # Update frequencies and periods
        self.freqs = {
            'pos': FREQ_POSITION,
            'vel': FREQ_VELOCITY,
            'att': FREQ_ATTITUDE,
            'rate': FREQ_ANGLE_RATE
        }
        self.dts = {k: 1.0/v for k, v in self.freqs.items()}
        
        # Last update times
        self.last_updates = {
            'pos': 0.0,
            'vel': 0.0,
            'att': 0.0,
            'rate': 0.0
        }
        
        # Initialize Controllers
        self.pos_pid = PosPID(POS_KP, self.dts['pos'])
        self.vel_pid = VelPID(VEL_KP, VEL_KI, VEL_KD, self.dts['vel'])
        self.att_pid = AttitudePID3D(ATT_KP, self.dts['att'])
        self.rate_pid = AngVelPID(RATE_KP, RATE_KI, RATE_KD, self.dts['rate'])
        
        # State Targets
        self.target_position = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).view(3, 1)
        self.target_velocity = torch.zeros((3, 1), dtype=torch.float32)
        self.target_attitude = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).view(4, 1) # [w, x, y, z]
        self.target_rate = torch.zeros((3, 1), dtype=torch.float32)
        self.target_yaw = 0.0 # Radians
        
        # Outputs
        self.total_thrust_vector = torch.zeros((3, 1), dtype=torch.float32) # World frame force
        self.torque_command = torch.zeros((3, 1), dtype=torch.float32)      # Body frame torque
        self.motor_thrusts = np.zeros(4) # Keep as numpy for mixer if mixer is numpy
        
        # For logging/debugging
        self.mixer_output_log = np.zeros(6) # [Fx, Fy, Fz, Mx, My, Mz]

    def reset(self):
        """Resets all sub-controllers and state."""
        self.pos_pid.reset()
        self.vel_pid.reset()
        self.att_pid.reset()
        self.rate_pid.reset()
        self.sim_time = 0.0
        for k in self.last_updates:
            self.last_updates[k] = 0.0

    def set_target_position(self, pos):
        self.target_position = torch.tensor(pos, dtype=torch.float32).view(3, 1)

    def set_target_yaw(self, yaw_rad):
        self.target_yaw = yaw_rad

    def update(self, sim_time: float) -> None:
        """
        Main control loop for the UAV.
        Updates position, velocity, attitude, and rate setpoints based on the current trajectory.
        
        Args:
            sim_time (float): Current simulation time.
        """
        self.sim_time = sim_time
        
        with torch.no_grad():
            # Get current state from model (returns numpy, we convert to explicit 2D tensors)
            pos_np, vel_np, att_np, rate_np = self.uav.get_uav_state()
            
            UAV_position = torch.tensor(pos_np, dtype=torch.float32).view(3, 1)
            UAV_velocity = torch.tensor(vel_np, dtype=torch.float32).view(3, 1)
            UAV_current_att = torch.tensor(att_np, dtype=torch.float32).view(4, 1) # [w, x, y, z]
            UAV_rate = torch.tensor(rate_np, dtype=torch.float32).view(3, 1)
    
            # 1. Position Controller (50Hz)
            if sim_time >= self.last_updates['pos'] + self.dts['pos']:
                self.target_velocity = self.pos_pid.update(self.target_position, UAV_position)
                self.last_updates['pos'] = sim_time
    
            # 2. Velocity Controller (50Hz)
            if sim_time >= self.last_updates['vel'] + self.dts['vel']:
                # Desired acceleration
                acc_cmd = self.vel_pid.update(self.target_velocity, UAV_velocity)
                
                # Gravity Compensation (World Frame: +Z opposes gravity)
                gravity_comp = torch.tensor([[0.0], [0.0], [GRAVITY]], dtype=torch.float32)
                
                # Total desired acceleration in World Frame
                total_acc = acc_cmd + gravity_comp
                
                # --- FIX: Prevent inverting the drone during fast descent ---
                # A standard quadrotor cannot generate downward thrust without flipping.
                # To prevent it from trying to flip when commanded to descend rapidly,
                # we constrain the Z-component of the total acceleration to be strictly positive.
                if total_acc[2, 0] < 0.1:
                    total_acc[2, 0] = 0.1
                
                # Calculate Total Thrust Vector (World Frame)
                self.total_thrust_vector = self.mass * total_acc
                
                # Calculate Desired Attitude from Thrust Vector
                self.target_attitude = self._calculate_desired_attitude(self.total_thrust_vector, self.target_yaw)
                
                self.last_updates['vel'] = sim_time
    
            # 3. Attitude Controller (250Hz)
            if sim_time >= self.last_updates['att'] + self.dts['att']:
                self.target_rate = self.att_pid.update(self.target_attitude, UAV_current_att)
                self.last_updates['att'] = sim_time
    
            # 4. Angular Rate Controller (1000Hz)
            if sim_time >= self.last_updates['rate'] + self.dts['rate']:
                self.torque_command = self.rate_pid.update(self.target_rate, UAV_rate)
                
                # Apply Control
                self._apply_controls(UAV_current_att)
                
                self.last_updates['rate'] = sim_time

    def print_state(self):
        return self.uav.print_uav_state()

    def _calculate_desired_attitude(self, thrust_vector: torch.Tensor, yaw_target: float) -> torch.Tensor:
        """
        Calculates the desired attitude quaternion to align the body Z-axis 
        with the desired thrust vector, while maintaining the target yaw.
        
        Args:
            thrust_vector (torch.Tensor): Shape: (3, 1), Dtype: torch.float32, Meaning: Desired thrust vector in world frame
            yaw_target (float): Yaw angle in radians
        
        Returns:
            torch.Tensor: Desired quaternion [w, x, y, z], Shape: (4, 1), Dtype: torch.float32
        """
        with torch.no_grad():
            norm_thrust = torch.norm(thrust_vector)
            z_b = thrust_vector / (norm_thrust + 1e-6)
            
            yaw_axis = torch.tensor([[-torch.sin(torch.tensor(yaw_target))], [torch.cos(torch.tensor(yaw_target))], [0.0]], dtype=torch.float32)
            
            # Cross product requires 1D or flattened for torch.cross in basic usage
            x_b = torch.cross(yaw_axis.view(-1), z_b.view(-1), dim=0).view(3, 1)
            x_b_norm = torch.norm(x_b)
            
            if x_b_norm < 1e-6:
                x_b = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float32)
            else:
                x_b = x_b / x_b_norm
                
            y_b = torch.cross(z_b.view(-1), x_b.view(-1), dim=0).view(3, 1)
            y_b = y_b / torch.norm(y_b)
            
            # Rotation Matrix [x_b, y_b, z_b]
            rot_mat = torch.cat((x_b, y_b, z_b), dim=1) # Shape: (3, 3)
            
            # Safe numpy conversion for quaternion library
            rot_mat_np = rot_mat.cpu().numpy()
            q_np = quaternion.from_rotation_matrix(rot_mat_np)
            return torch.tensor([[q_np.w], [q_np.x], [q_np.y], [q_np.z]], dtype=torch.float32)

    def _apply_controls(self, current_att: torch.Tensor) -> None:
        """
        Maps total thrust vector and torque commands to motor speeds.
        
        Args:
            current_att (torch.Tensor): Current attitude quaternion (w, x, y, z), Shape: (4, 1), Dtype: torch.float32
        """
        with torch.no_grad():
            # Reason: quaternion library expects 1D array.
            # Current dimension: (4, 1) -> (4,)
            q = np.quaternion(*current_att.squeeze().cpu().numpy())
            rot_mat_np = quaternion.as_rotation_matrix(q)
            rot_mat = torch.tensor(rot_mat_np, dtype=torch.float32)
            
            # Transform World Thrust -> Body Frame
            # F_body = R^T * F_world
            force_body = rot_mat.T @ self.total_thrust_vector
            
            # Convert back to numpy for logging and external models
            # Reason: uav_model and mixer expect numpy arrays.
            # Current dimensions: (3, 1) -> (3,)
            f_body_np = force_body.squeeze().cpu().numpy()
            t_body_np = self.torque_command.squeeze().cpu().numpy()
            
            self.mixer_output_log[:3] = f_body_np
            self.mixer_output_log[3:] = t_body_np
            
            # self.uav.set_actuators(f_body_np, t_body_np) # Enabled direct force actuation for debug
            
            # --- FIX: Saturation of thrust ---
            # Drones can only generate positive thrust in the body Z direction.
            # We must not allow negative thrust (which would imply pulling the drone downwards).
            from ..config import CT, MAX_MOTOR_SPEED_KRPM
            max_total_thrust = 4 * CT * (MAX_MOTOR_SPEED_KRPM ** 2)
            # Make sure we use the positive Z thrust from the body frame (which points upwards relative to drone)
            # thrust_mag is Fz in body frame.
            thrust_mag = np.clip(f_body_np[2], 0.0, max_total_thrust)
            
            # Use the SE3 style mixing matrix (returns squared speeds in krpm^2)
            motor_speeds_sq = self.mixer.calculate(thrust_mag, t_body_np[0], t_body_np[1], t_body_np[2])
            
            # Apply squared motor speeds to MuJoCo rotor actuators
            self.uav.set_motor_speeds(motor_speeds_sq)
            
            # For logging actual thrust per motor: T = Ct * w^2
            self.motor_thrusts = motor_speeds_sq * self.mixer.Ct

    def get_log_data(self):
        """Returns data for logging."""
        UAV_pos, UAV_vel, UAV_att, UAV_rate = self.uav.get_uav_state()
        return (
            self.sim_time, UAV_pos, UAV_vel, UAV_att, UAV_rate,
            self.target_position.squeeze().cpu().numpy(), 
            self.target_velocity.squeeze().cpu().numpy(), 
            self.target_attitude.squeeze().cpu().numpy(), 
            self.target_rate.squeeze().cpu().numpy(),
            self.motor_thrusts, self.mixer_output_log
        )
