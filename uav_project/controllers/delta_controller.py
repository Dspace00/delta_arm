import torch
import numpy as np
import math
from uav_project.utils.DeltaKinematics import DeltaKinematics

class DeltaController:
    """
    Controller for the Delta robot manipulator using Jacobian-based velocity control.
    """
    def __init__(self, uav_model, control_freq=100.0, control_mode='position'):
        """
        Args:
            uav_model: Instance of UAVModel.
            control_freq: Control frequency in Hz.
            control_mode: 'position' or 'velocity'.
        """
        self.uav = uav_model
        self.control_freq = control_freq
        self.control_mode = control_mode
        self.dt = 1.0 / control_freq
        
        # Delta Robot Parameters
        self.L = 0.1      # Upper arm length (rod_b)
        self.l = 0.2      # Lower arm length (rod_ee)
        self.R = 0.074577 # Base radius (r_b)
        self.r = 0.02495  # End-effector radius (r_ee)
        
        # Kinematics helper
        self.kinematics = DeltaKinematics(self.L, self.l, self.R, self.r)
        
        # Control gains (PD)
        self.kp = 2.0
        self.kd = 1.0
        
        # Trajectory parameters
        self.traj_radius = 0.12
        self.traj_z = -0.18
        self.traj_period = 8.0 # seconds for one circle
        
        self.last_update_time = 0.0
        self.current_des_pos_log = np.zeros(3)
        self.current_actual_pos_log = np.zeros(3)
        
    def get_circular_trajectory(self, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates desired position and velocity for a circular trajectory.
        
        Args:
            t (float): Current time.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - des_pos (torch.Tensor): Desired position, Shape: (3, 1), Dtype: torch.float32
                - des_vel (torch.Tensor): Desired velocity, Shape: (3, 1), Dtype: torch.float32
        """
        with torch.no_grad():
            # Use torch constants and math functions
            center = torch.tensor([[0.0], [0.0], [self.traj_z]], dtype=torch.float32)
            omega = 2 * torch.pi / self.traj_period
            phase = omega * t
            
            # Position
            x = self.traj_radius * torch.cos(torch.tensor(phase))
            y = self.traj_radius * torch.sin(torch.tensor(phase))
            z = self.traj_z
            
            # Velocity
            vx = -self.traj_radius * omega * torch.sin(torch.tensor(phase))
            vy = self.traj_radius * omega * torch.cos(torch.tensor(phase))
            vz = 0.0
            
            des_pos = torch.tensor([[x], [y], [z]], dtype=torch.float32)
            des_vel = torch.tensor([[vx], [vy], [vz]], dtype=torch.float32)
            
            return des_pos, des_vel

    def update(self, sim_time: float) -> None:
        """
        Main update loop. Should be called every simulation step, 
        but logic executes only at control_freq.
        
        Args:
            sim_time (float): Current simulation time.
        """
        if sim_time >= self.last_update_time + self.dt:
            
            # Get current state from numpy sensors and convert to PyTorch (3, 1)
            pos_np = self.uav.get_ee_sensor_pos()
            vel_np = self.uav.get_ee_sensor_lin_vel()
            
            current_pos = torch.tensor(pos_np, dtype=torch.float32).view(3, 1)
            current_vel = torch.tensor(vel_np, dtype=torch.float32).view(3, 1)
            
            # Get desired state
            des_pos, des_vel = self.get_circular_trajectory(sim_time)
            
            # Update log variables
            self.current_des_pos_log = des_pos.squeeze().cpu().numpy()
            self.current_actual_pos_log = pos_np
            
            # Temporary print for tracking error
            pos_err = self.current_des_pos_log - self.current_actual_pos_log
            # print(f"Time: {sim_time:.2f} | Delta EE Error (m): X={pos_err[0]:.4f}, Y={pos_err[1]:.4f}, Z={pos_err[2]:.4f}")
            
            # 3. Calculate Controls
            if self.control_mode == 'position':
                # Convert pos back to numpy for IK library
                # Reason: DeltaKinematics.ik expects a Tensor now
                # Current Dimension: des_pos shape (3, 1) -> (3,)
                des_pos_tensor = des_pos.squeeze()
                joint_angles_deg = self.kinematics.ik(des_pos_tensor)
                
                if isinstance(joint_angles_deg, int) and joint_angles_deg == -1:
                    print(f"Warning: IK failed for target {des_pos_tensor}")
                else:
                    joint_angles_rad = torch.deg2rad(joint_angles_deg)
                    self.uav.set_delta_motor_positions(joint_angles_rad)
                    
            elif self.control_mode == 'velocity':
                # Velocity Control: Jacobian-based
                motor_vels = self.calculate_motor_velocities(current_pos, current_vel, des_pos, des_vel)
                # Convert to numpy for mujoco interaction
                # Reason: uav_model.set_delta_motor_velocities expects a 1D numpy array (3,)
                # Current Dimension: motor_vels shape (3, 1) -> (3,)
                motor_vels_np = motor_vels.squeeze().cpu().numpy()
                self.uav.set_delta_motor_velocities(motor_vels_np)

            self.last_update_time = sim_time

    def calculate_motor_velocities(self, pos: torch.Tensor, vel: torch.Tensor, des_pos: torch.Tensor, des_vel: torch.Tensor) -> torch.Tensor:
        """
        Calculates motor angular velocities using Jacobian-based PD control.
        
        Args:
            pos (torch.Tensor): Current position, Shape: (3, 1), Dtype: torch.float32
            vel (torch.Tensor): Current velocity, Shape: (3, 1), Dtype: torch.float32
            des_pos (torch.Tensor): Desired position, Shape: (3, 1), Dtype: torch.float32
            des_vel (torch.Tensor): Desired velocity, Shape: (3, 1), Dtype: torch.float32
            
        Returns:
            torch.Tensor: Desired motor angular velocities, Shape: (3, 1), Dtype: torch.float32
        """
        with torch.no_grad():
            # PD Control for Velocity Correction
            err_pos = des_pos - pos
            err_vel = des_vel - vel
            
            vel_correct = self.kp * err_pos + self.kd * err_vel
            vel_control = des_vel + vel_correct
            
            # Convert to numpy temporarily for IK
            # Reason: DeltaKinematics.ik expects 1D array
            # Current Dimension: pos shape (3, 1) -> (3,)
            pos_np = pos.squeeze().cpu().numpy()
            current_angles_deg = self.kinematics.ik(pos_np)
            if isinstance(current_angles_deg, int) and current_angles_deg == -1:
                 return torch.zeros((3, 1), dtype=torch.float32)
            
            current_angles = torch.tensor(np.deg2rad(current_angles_deg), dtype=torch.float32)
            
            # Constants
            L_upper = self.L
            L_lower = self.l
            R_val = self.R
            r_val = self.r
            
            def phi_matrix(i):
                angle = torch.tensor(120 * i * torch.pi / 180, dtype=torch.float32)
                c = torch.cos(angle)
                s = torch.sin(angle)
                return torch.tensor([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=torch.float32)

            def alpha_vector(q_i):
                return torch.tensor([[torch.cos(q_i)], [0], [torch.sin(q_i)]], dtype=torch.float32)

            def d_alpha_dq(q_i):
                return torch.tensor([[-torch.sin(q_i)], [0], [torch.cos(q_i)]], dtype=torch.float32)

            def e_vector(q_i, i):
                alpha_i = alpha_vector(q_i)
                base_vector = torch.tensor([[R_val - r_val], [0], [0]], dtype=torch.float32)
                return base_vector + L_upper * alpha_i

            def beta_vector(p, q_i, i):
                phi_i = phi_matrix(i)
                p_Ei_e = phi_i @ p
                e_i = e_vector(q_i, i)
                return (p_Ei_e - e_i) / L_lower

            M_rows = []
            V_diag = []

            for i in range(3):
                q_i = current_angles[i]
                beta_i = beta_vector(pos, q_i, i)
                phi_i = phi_matrix(i)
                
                # M_row: 1x3
                M_row = beta_i.T @ phi_i
                M_rows.append(M_row)
                
                d_alpha = d_alpha_dq(q_i)
                # V_element: 1x1
                V_element = beta_i.T @ d_alpha
                V_diag.append(V_element.view(-1))

            # M: 3x3, V: 3x3
            M = torch.cat(M_rows, dim=0)
            V = torch.diag(torch.cat(V_diag))

            try:
                V_inv = torch.linalg.inv(V)
                q_dot = (1.0 / L_upper) * V_inv @ M @ vel_control
            except RuntimeError:
                # Handle singularity
                V_inv = torch.linalg.pinv(V)
                q_dot = (1.0 / L_upper) * V_inv @ M @ vel_control

            return q_dot
