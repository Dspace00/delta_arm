"""
Motor mixing logic for converting forces/torques to motor speeds.
"""
import numpy as np
from ..config import CT, CM, ARM_LENGTH, MAX_MOTOR_SPEED_KRPM

class Mixer:
    """
    Handles mixing of control inputs (Thrust + Torques) to motor outputs (RPM/Force).
    """
    def __init__(self):
        self.Ct = CT
        self.Cm = CM
        self.L = ARM_LENGTH
        self.max_speed = MAX_MOTOR_SPEED_KRPM
        
        # ---------------------------------------------------------
        # NEW MIXING MATRIX: Maps motor squared speeds (krpm^2) to forces/torques
        # ---------------------------------------------------------
        # Input: [w1^2, w2^2, w3^2, w4^2]^T  (Motor squared speeds in krpm^2)
        # Output: [Fz, Mx, My, Mz]^T 
        #
        # Fz = Ct * (w1^2 + w2^2 + w3^2 + w4^2)
        # Mx = Ct * L * (w1^2 - w2^2 - w3^2 + w4^2)
        # My = Ct * L * (-w1^2 - w2^2 + w3^2 + w4^2)
        # Mz = Cm * (-w1^2 + w2^2 - w3^2 + w4^2)
        
        # Mapping Matrix M: [Fz, Mx, My, Mz]^T = M @ [w1^2, w2^2, w3^2, w4^2]^T
        self.mat = np.array([
            [self.Ct, self.Ct, self.Ct, self.Ct],
            [self.Ct*self.L, -self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L],
            [-self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L, self.Ct*self.L],
            [-self.Cm, self.Cm, -self.Cm, self.Cm]
        ])
        
        # Pseudo-inverse of M
        self.inv_mat = np.linalg.pinv(self.mat)

    def calculate(self, thrust: float, mx: float, my: float, mz: float) -> np.ndarray:
        """
        Calculates motor squared speeds (krpm^2) from desired total thrust and torques 
        using SE3-style priority desaturation logic.
        
        Priority: Thrust > Roll/Pitch > Yaw
        
        Args:
            thrust (float): Total desired thrust in body Z (N).
            mx (float): Desired moment around body X (Nm).
            my (float): Desired moment around body Y (Nm).
            mz (float): Desired moment around body Z (Nm).
            
        Returns:
            np.ndarray: Motor squared speeds in krpm^2, Shape: (4,)
        """
        Mx, My = mx, my
        Mz = 0.0  # Initially, do not allocate Yaw torque
        
        # Reason: inv_mat is (4,4), control_input must be (4, 1) to maintain 2D structural shape
        control_input = np.array([[thrust], [Mx], [My], [Mz]])
        
        # 1. First Allocation (Only Fz, Mx, My)
        # t_base shape: (4, 1), representing w^2
        t_base = self.inv_mat @ control_input
        
        # 2. Check X/Y Saturation
        max_value = np.max(t_base)
        min_value = np.min(t_base)
        
        # Reference w^2 is just total thrust / (4 * Ct)
        ref_value = thrust / (4.0 * self.Ct)
        
        # Max allowed w^2
        max_w_sq = self.max_speed ** 2
        
        max_trim_scale = 1.0
        min_trim_scale = 1.0
        
        if max_value > max_w_sq:
            denom = max_value - ref_value
            max_trim_scale = (max_w_sq - ref_value) / denom if denom != 0 else 0.0
            
        if min_value < 0.0:
            denom = ref_value - min_value
            min_trim_scale = ref_value / denom if denom != 0 else 0.0
            
        scale = min(max_trim_scale, min_trim_scale)
        
        # 3. Apply scale to X/Y torques
        Mx = Mx * scale
        My = My * scale
        
        # Re-calculate with scaled torques
        control_input = np.array([[thrust], [Mx], [My], [Mz]])
        t_base = self.inv_mat @ control_input
        
        # 4. Handle Yaw (Mz) Allocation
        if scale < 1.0:
            # X/Y allocation has saturated the motors. Discard Yaw (Mz) entirely.
            # Final safety clip
            t_final = np.clip(t_base, 0.0, max_w_sq)
        else:
            # We have headroom, try to allocate Yaw
            Mz = mz
            control_input_withz = np.array([[thrust], [Mx], [My], [Mz]])
            t_withz = self.inv_mat @ control_input_withz
            
            # Check saturation caused by Z
            max_val_z = np.max(t_withz)
            min_val_z = np.min(t_withz)
            
            # Find the motor that caused saturation and its base value
            max_idx = np.argmax(t_withz)
            min_idx = np.argmin(t_withz)
            
            max_trim_scale_z = 1.0
            min_trim_scale_z = 1.0
            
            if max_val_z > max_w_sq:
                denom = max_val_z - t_base[max_idx, 0]
                max_trim_scale_z = (max_w_sq - t_base[max_idx, 0]) / denom if denom != 0 else 0.0
                
            if min_val_z < 0.0:
                denom = t_base[min_idx, 0] - min_val_z
                min_trim_scale_z = t_base[min_idx, 0] / denom if denom != 0 else 0.0
                
            scale_z = min(max_trim_scale_z, min_trim_scale_z)
            
            Mz = Mz * scale_z
            
            # Final calculation
            control_input_final = np.array([[thrust], [Mx], [My], [Mz]])
            t_final = self.inv_mat @ control_input_final
            t_final = np.clip(t_final, 0.0, max_w_sq)

        # Reason: Controller code currently expects 1D array (4,) for assignment to MuJoCo
        # Dimension: (4, 1) -> (4,)
        return t_final.squeeze()

    def simple_mix(self, total_thrust, torques):
        """
        A simpler mixing strategy that returns motor thrusts (N) instead of RPM.
        Used for simplified simulation control.
        """
        # Note: This matrix must match the one used in `apply_motor_thrusts` in original code
        # The original code uses:
        # [c, c, c, c]
        # [c*L, -c*L, -c*L, c*L]
        # [-c*L, -c*L, c*L, c*L]
        # [-b, b, -b, b]
        # This matches self.mat if c=Ct, b=Cd.
        
        forces_torques = np.array([total_thrust, torques[0], torques[1], torques[2]])
        
        # Solve for squared speeds first (or pseudo-thrusts if we consider T ~ w^2 directly)
        # If we want motor THRUSTS (N), we need to invert the relationship carefully.
        # But for simplified control where we just want to distribute forces:
        
        # Let's assume the mixing matrix relates motor THRUSTS to body wrench directly:
        # F_total = sum(Fi)
        # Mx = L * (F0 + F3 - F1 - F2)  (Based on signs in original matrix)
        # ...
        
        # Let's reconstruct the matrix for Force -> Wrench
        # Note: Original code uses self.thrust_coeff=1 for simple mixing
        c = 1.0
        b = self.Cm / self.Ct  # torque_coeff
        L = self.L
        
        # Matrix mapping [F0, F1, F2, F3] -> [F_total, Mx, My, Mz]
        # Indices: 0:FrontLeft, 1:FrontRight, 2:BackRight, 3:BackLeft
        # Check signs from original:
        # F_total: + + + +
        # Mx: + - - +  => F0, F3 pos; F1, F2 neg
        # My: - - + +  => F2, F3 pos; F0, F1 neg
        # Mz: - + - +  => F1, F3 pos; F0, F2 neg
        
        mixing_matrix = np.array([
            [c, c, c, c],                   # Total Thrust
            [c*L, -c*L, -c*L, c*L],         # Mx
            [-c*L, -c*L, c*L, c*L],         # My
            [-b, b, -b, b]                  # Mz
        ])
        
        motor_thrusts = np.linalg.inv(mixing_matrix) @ forces_torques
        return motor_thrusts
