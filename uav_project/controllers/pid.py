"""
PID Controller implementations for UAV control.
Includes Position (P), Velocity (PID), Attitude (P-Quaternion), and Angular Rate (PID) controllers.
"""

import torch
import numpy as np
import quaternion # numpy-quaternion for external interop if needed, but we will mostly use torch

class BasePID:
    """Base class for PID controllers."""
    def __init__(self, output_limits=(-float('inf'), float('inf'))):
        self.output_limits = output_limits

    def clip_output(self, output: torch.Tensor) -> torch.Tensor:
        """Clips the output to defined limits."""
        if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
            return torch.clamp(output, self.output_limits[0], self.output_limits[1])
        return output
    
    def reset(self):
        """Resets the internal state of the controller."""
        pass


class PosPID(BasePID):
    """
    Position Controller (P-only).
    Outputs desired velocity based on position error.
    """
    def __init__(self, kp: float, dt: float, output_limits=(-float('inf'), float('inf'))):
        super().__init__(output_limits)
        self.kp = torch.tensor(kp, dtype=torch.float32).view(-1, 1) if hasattr(kp, '__len__') else torch.tensor(kp, dtype=torch.float32)
        self.dt = dt

    def update(self, setpoint: torch.Tensor, current_value: torch.Tensor) -> torch.Tensor:
        """
        Updates the controller.
        
        Args:
            setpoint (torch.Tensor): Target position, Shape: (3, 1), Dtype: torch.float32.
            current_value (torch.Tensor): Current position, Shape: (3, 1), Dtype: torch.float32.
            
        Returns:
            torch.Tensor: Target velocity, Shape: (3, 1), Dtype: torch.float32.
        """
        with torch.no_grad():
            error = setpoint - current_value
            output = self.kp * error
            return self.clip_output(output)


class VelPID(BasePID):
    """
    Velocity Controller (PID).
    Outputs desired acceleration (or thrust vector components) based on velocity error.
    """
    def __init__(self, kp: float, ki: float, kd: float, dt: float, output_limits=(-float('inf'), float('inf'))):
        super().__init__(output_limits)
        self.kp = torch.tensor(kp, dtype=torch.float32).view(-1, 1) if hasattr(kp, '__len__') else torch.tensor(kp, dtype=torch.float32)
        self.ki = torch.tensor(ki, dtype=torch.float32).view(-1, 1) if hasattr(ki, '__len__') else torch.tensor(ki, dtype=torch.float32)
        self.kd = torch.tensor(kd, dtype=torch.float32).view(-1, 1) if hasattr(kd, '__len__') else torch.tensor(kd, dtype=torch.float32)
        self.dt = dt
        
        self.integral = torch.zeros((3, 1), dtype=torch.float32)
        self.previous_error = torch.zeros((3, 1), dtype=torch.float32)
        self.last_value = torch.zeros((3, 1), dtype=torch.float32)

    def update(self, setpoint: torch.Tensor, current_value: torch.Tensor) -> torch.Tensor:
        """
        Updates the controller.
        
        Args:
            setpoint (torch.Tensor): Target velocity, Shape: (3, 1), Dtype: torch.float32.
            current_value (torch.Tensor): Current velocity, Shape: (3, 1), Dtype: torch.float32.
            
        Returns:
            torch.Tensor: Desired acceleration, Shape: (3, 1), Dtype: torch.float32.
        """
        with torch.no_grad():
            error = setpoint - current_value
            
            # Proportional term
            p_term = self.kp * error
            
            # Integral term
            self.integral += error * self.dt
            i_term = self.ki * self.integral
            
            # Derivative term (on measurement to avoid derivative kick)
            v_dot = (current_value - self.last_value) / self.dt
            self.last_value = current_value.clone()
            d_term = self.kd * (0.0 - v_dot)
            
            output = p_term + i_term + d_term
            self.previous_error = error.clone()
            
            return self.clip_output(output)
    
    def reset(self):
        self.integral.zero_()
        self.previous_error.zero_()
        self.last_value.zero_()


class AttitudePID3D(BasePID):
    """
    Attitude Controller (P-only on Quaternion Error).
    Outputs desired angular rates based on attitude error.
    """
    def __init__(self, kp: float, dt: float, output_limits=(-float('inf'), float('inf'))):
        super().__init__(output_limits)
        self.kp = torch.tensor(kp, dtype=torch.float32).view(-1, 1) if hasattr(kp, '__len__') else torch.tensor(kp, dtype=torch.float32)
        self.dt = dt

    def update(self, setpoint: torch.Tensor, current_value: torch.Tensor) -> torch.Tensor:
        """
        Calculates the attitude PID control output using quaternions.
        
        Args:
            setpoint (torch.Tensor): Desired attitude quaternion (w, x, y, z), Shape: (4, 1), Dtype: torch.float32
            current_value (torch.Tensor): Current attitude quaternion (w, x, y, z), Shape: (4, 1), Dtype: torch.float32
            
        Returns:
            torch.Tensor: Control torque command, Shape: (3, 1), Dtype: torch.float32
        """
        with torch.no_grad():
            # Ensure safe numpy conversions for quaternion math, keeping 2D logic
            # Alternatively, do quaternion math purely in PyTorch
            # q_err = q_target * q_current_inverse
            
            # For simplicity, convert to numpy-quaternion, compute error, convert back
            # Reason: quaternion library is robust for quaternion math. 
            # Current dimension: (4, 1) -> (4,) for quat initialization
            tar_q = np.quaternion(*setpoint.squeeze().cpu().numpy())
            cur_q = np.quaternion(*current_value.squeeze().cpu().numpy())
            
            # Correct attitude error in body frame: q_err = q_cur^{-1} * q_tar
            cur_conj = cur_q.conjugate()
            quat_error = cur_conj * tar_q
            
            # Extract vector part
            error_np = np.array([quat_error.x, quat_error.y, quat_error.z]).reshape(3, 1)
            
            if quat_error.w < 0:
                error_np = -error_np
                
            error = torch.tensor(error_np, dtype=torch.float32, device=setpoint.device)
            output = self.kp * error
            
            return self.clip_output(output)


class AngVelPID(BasePID):
    """
    Angular Velocity Controller (PID).
    Outputs control torques based on angular rate error.
    """
    def __init__(self, kp: float, ki: float, kd: float, dt: float, output_limits=(-float('inf'), float('inf'))):
        super().__init__(output_limits)
        self.kp = torch.tensor(kp, dtype=torch.float32).view(-1, 1) if hasattr(kp, '__len__') else torch.tensor(kp, dtype=torch.float32)
        self.ki = torch.tensor(ki, dtype=torch.float32).view(-1, 1) if hasattr(ki, '__len__') else torch.tensor(ki, dtype=torch.float32)
        self.kd = torch.tensor(kd, dtype=torch.float32).view(-1, 1) if hasattr(kd, '__len__') else torch.tensor(kd, dtype=torch.float32)
        self.dt = dt
        
        self.integral = torch.zeros((3, 1), dtype=torch.float32)
        self.previous_error = torch.zeros((3, 1), dtype=torch.float32)
        self.last_value = torch.zeros((3, 1), dtype=torch.float32)

    def update(self, setpoint: torch.Tensor, current_value: torch.Tensor) -> torch.Tensor:
        """
        Updates the controller.
        
        Args:
            setpoint (torch.Tensor): Target angular rates, Shape: (3, 1), Dtype: torch.float32.
            current_value (torch.Tensor): Current angular rates, Shape: (3, 1), Dtype: torch.float32.
            
        Returns:
            torch.Tensor: Control torques [Mx, My, Mz], Shape: (3, 1), Dtype: torch.float32.
        """
        with torch.no_grad():
            error = setpoint - current_value
            
            p_term = self.kp * error
            
            self.integral += error * self.dt
            i_term = self.ki * self.integral
            
            v_dot = (current_value - self.last_value) / self.dt
            self.last_value = current_value.clone()
            d_term = self.kd * (0.0 - v_dot)
            
            output = p_term + i_term + d_term
            self.previous_error = error.clone()
            
            return self.clip_output(output)
        
    def reset(self):
        self.integral.zero_()
        self.previous_error.zero_()
        self.last_value.zero_()
