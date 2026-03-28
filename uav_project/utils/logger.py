"""
Logger utility for recording simulation data and plotting results.
"""

import os
import platform
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation as R

# Configure matplotlib backend for headless environments (SSH constraints)
matplotlib.use('Agg')

import matplotlib.pyplot as plt

class Logger:
    """
    Records simulation data and generates plots.
    Supports both UAV and Delta robot simulations.
    """
    def __init__(self):
        self.history = {
            # --- UAV Fields ---
            'time': [],
            'position': [],
            'velocity': [],
            'target_velocity': [],
            'euler': [],
            'target_position': [],
            'target_euler': [],
            'angle_rate': [],
            'target_angle_rate': [],
            'motor_thrusts': [],
            'motor_mix': [],
            # --- Delta Fields (existing) ---
            'delta_des_pos': [],
            'delta_actual_pos': [],
            # --- Delta Fields (new) ---
            'delta_des_vel': [],
            'delta_actual_vel': [],
            'delta_motor_angles': [],
            'delta_motor_velocities': [],
            'delta_ik_success': [],
            'delta_workspace_status': [],
        }

    def log(self, time, position, velocity, attitude_quat, angle_rate, 
            target_pos, target_vel, target_att_quat, target_rate, 
            motor_thrusts, mixer_outputs, delta_des_pos=None, delta_actual_pos=None):
        """
        Appends a new data point to history.
        
        Args:
            time: Current simulation time.
            position: Current position [x, y, z].
            velocity: Current velocity [vx, vy, vz].
            attitude_quat: Current attitude quaternion [w, x, y, z].
            angle_rate: Current angular velocity [p, q, r].
            target_pos: Target position.
            target_vel: Target velocity.
            target_att_quat: Target attitude quaternion.
            target_rate: Target angular rate.
            motor_thrusts: Array of motor thrusts.
            mixer_outputs: Mixer outputs (Forces + Torques).
            delta_des_pos: Delta robot desired position.
            delta_actual_pos: Delta robot actual position.
        """
        # Convert quaternions to Euler angles for logging/plotting (ZYX order)
        # Note: input quat is [w, x, y, z] (scalar first)
        
        # Current Euler
        try:
            r = R.from_quat([attitude_quat[1], attitude_quat[2], attitude_quat[3], attitude_quat[0]]) # Expects [x, y, z, w]
            euler = r.as_euler('ZYX', degrees=True)
        except Exception:
            euler = np.zeros(3)

        # Target Euler
        try:
            # Check if target_att_quat is a numpy-quaternion object or array
            if hasattr(target_att_quat, 'w'):
                t_q = [target_att_quat.x, target_att_quat.y, target_att_quat.z, target_att_quat.w]
            else:
                # Assuming [w, x, y, z]
                t_q = [target_att_quat[1], target_att_quat[2], target_att_quat[3], target_att_quat[0]]
            
            r_t = R.from_quat(t_q)
            target_euler = r_t.as_euler('ZYX', degrees=True)
        except Exception:
            target_euler = np.zeros(3)

        self.history['time'].append(time)
        self.history['position'].append(np.array(position).copy())
        self.history['velocity'].append(np.array(velocity).copy())
        self.history['target_velocity'].append(np.array(target_vel).copy())
        self.history['euler'].append(euler.copy())
        self.history['target_position'].append(np.array(target_pos).copy())
        self.history['target_euler'].append(target_euler.copy())
        self.history['angle_rate'].append(np.array(angle_rate).copy())
        self.history['target_angle_rate'].append(np.array(target_rate).copy())
        self.history['motor_thrusts'].append(np.array(motor_thrusts).copy())
        self.history['motor_mix'].append(np.array(mixer_outputs).copy())
        
        if delta_des_pos is not None:
            self.history['delta_des_pos'].append(np.array(delta_des_pos).copy())
        else:
            self.history['delta_des_pos'].append(None)
            
        if delta_actual_pos is not None:
            self.history['delta_actual_pos'].append(np.array(delta_actual_pos).copy())
        else:
            self.history['delta_actual_pos'].append(None)

    def plot_results(self, save_path='simulation_results.png'):
        """
        Plots the recorded history with an optimized layout for cascading PID debugging.
        """
        try:
            time_array = np.array(self.history['time'])
            if len(time_array) == 0:
                print("No data to plot.")
                return

            position = np.array(self.history['position'])
            target_position = np.array(self.history['target_position'])
            
            vel = np.array(self.history['velocity'])
            target_vel = np.array(self.history['target_velocity'])
            
            euler = np.array(self.history['euler'])
            target_euler = np.array(self.history['target_euler'])
            
            angle_rate = np.array(self.history['angle_rate'])
            target_angle_rate = np.array(self.history['target_angle_rate'])
            
            motor_thrusts = np.array(self.history['motor_thrusts'])
            motor_mix = np.array(self.history['motor_mix'])
            
            has_delta = len(self.history['delta_des_pos']) > 0 and self.history['delta_des_pos'][0] is not None
            rows = 6 if has_delta else 5
            
            fig = plt.figure(figsize=(20, 4 * rows))
            
            # --- Row 1: Position ---
            ax1 = fig.add_subplot(rows, 4, 1)
            ax1.plot(time_array, position[:, 0], 'r-', label='Act X')
            ax1.plot(time_array, target_position[:, 0], 'r--', label='Ref X')
            ax1.set_title('Position X (m)')
            ax1.grid(True); ax1.legend()
            
            ax2 = fig.add_subplot(rows, 4, 2)
            ax2.plot(time_array, position[:, 1], 'g-', label='Act Y')
            ax2.plot(time_array, target_position[:, 1], 'g--', label='Ref Y')
            ax2.set_title('Position Y (m)')
            ax2.grid(True); ax2.legend()
            
            ax3 = fig.add_subplot(rows, 4, 3)
            ax3.plot(time_array, position[:, 2], 'b-', label='Act Z')
            ax3.plot(time_array, target_position[:, 2], 'b--', label='Ref Z')
            ax3.set_title('Position Z (m)')
            ax3.grid(True); ax3.legend()
            
            ax4 = fig.add_subplot(rows, 4, 4)
            pos_error = np.linalg.norm(position - target_position, axis=1)
            ax4.plot(time_array, pos_error, 'k-')
            ax4.set_title('Position Error Norm (m)')
            ax4.grid(True)

            # --- Row 2: Velocity ---
            ax5 = fig.add_subplot(rows, 4, 5)
            ax5.plot(time_array, vel[:, 0], 'r-', label='Act Vx')
            ax5.plot(time_array, target_vel[:, 0], 'r--', label='Ref Vx')
            ax5.set_title('Velocity X (m/s)')
            ax5.grid(True); ax5.legend()
            
            ax6 = fig.add_subplot(rows, 4, 6)
            ax6.plot(time_array, vel[:, 1], 'g-', label='Act Vy')
            ax6.plot(time_array, target_vel[:, 1], 'g--', label='Ref Vy')
            ax6.set_title('Velocity Y (m/s)')
            ax6.grid(True); ax6.legend()
            
            ax7 = fig.add_subplot(rows, 4, 7)
            ax7.plot(time_array, vel[:, 2], 'b-', label='Act Vz')
            ax7.plot(time_array, target_vel[:, 2], 'b--', label='Ref Vz')
            ax7.set_title('Velocity Z (m/s)')
            ax7.grid(True); ax7.legend()
            
            ax8 = fig.add_subplot(rows, 4, 8)
            vel_error = np.linalg.norm(vel - target_vel, axis=1)
            ax8.plot(time_array, vel_error, 'k-')
            ax8.set_title('Velocity Error Norm (m/s)')
            ax8.grid(True)

            # --- Row 3: Attitude (Euler ZYX -> Roll, Pitch, Yaw) ---
            # Note: scipy as_euler('ZYX') returns [Yaw, Pitch, Roll]
            # euler[:, 0]=Yaw, euler[:, 1]=Pitch, euler[:, 2]=Roll
            ax9 = fig.add_subplot(rows, 4, 9)
            ax9.plot(time_array, euler[:, 2], 'r-', label='Act Roll')
            ax9.plot(time_array, target_euler[:, 2], 'r--', label='Ref Roll')
            ax9.set_title('Roll (deg)')
            ax9.grid(True); ax9.legend()
            
            ax10 = fig.add_subplot(rows, 4, 10)
            ax10.plot(time_array, euler[:, 1], 'g-', label='Act Pitch')
            ax10.plot(time_array, target_euler[:, 1], 'g--', label='Ref Pitch')
            ax10.set_title('Pitch (deg)')
            ax10.grid(True); ax10.legend()
            
            ax11 = fig.add_subplot(rows, 4, 11)
            ax11.plot(time_array, euler[:, 0], 'b-', label='Act Yaw')
            ax11.plot(time_array, target_euler[:, 0], 'b--', label='Ref Yaw')
            ax11.set_title('Yaw (deg)')
            ax11.grid(True); ax11.legend()
            
            ax12 = fig.add_subplot(rows, 4, 12)
            # Simple absolute error sum for Euler
            euler_error = np.linalg.norm(euler - target_euler, axis=1)
            ax12.plot(time_array, euler_error, 'k-')
            ax12.set_title('Euler Error Norm (deg)')
            ax12.grid(True)

            # --- Row 4: Angular Rate ---
            ax13 = fig.add_subplot(rows, 4, 13)
            ax13.plot(time_array, angle_rate[:, 0], 'r-', label='Act P')
            ax13.plot(time_array, target_angle_rate[:, 0], 'r--', label='Ref P')
            ax13.set_title('Roll Rate P (rad/s)')
            ax13.grid(True); ax13.legend()
            
            ax14 = fig.add_subplot(rows, 4, 14)
            ax14.plot(time_array, angle_rate[:, 1], 'g-', label='Act Q')
            ax14.plot(time_array, target_angle_rate[:, 1], 'g--', label='Ref Q')
            ax14.set_title('Pitch Rate Q (rad/s)')
            ax14.grid(True); ax14.legend()
            
            ax15 = fig.add_subplot(rows, 4, 15)
            ax15.plot(time_array, angle_rate[:, 2], 'b-', label='Act R')
            ax15.plot(time_array, target_angle_rate[:, 2], 'b--', label='Ref R')
            ax15.set_title('Yaw Rate R (rad/s)')
            ax15.grid(True); ax15.legend()
            
            ax16 = fig.add_subplot(rows, 4, 16)
            rate_error = np.linalg.norm(angle_rate - target_angle_rate, axis=1)
            ax16.plot(time_array, rate_error, 'k-')
            ax16.set_title('Angular Rate Error Norm')
            ax16.grid(True)

            # --- Row 5: Control Outputs (F, M) & Motor Thrusts & 3D Path ---
            ax17 = fig.add_subplot(rows, 4, 17)
            ax17.plot(time_array, motor_mix[:, 0], 'k-', label='Total Thrust (Fz)')
            ax17.set_title('Total Thrust F (N)')
            ax17.grid(True); ax17.legend()
            
            ax18 = fig.add_subplot(rows, 4, 18)
            ax18.plot(time_array, motor_mix[:, 3], 'r-', label='Mx')
            ax18.plot(time_array, motor_mix[:, 4], 'g-', label='My')
            ax18.plot(time_array, motor_mix[:, 5], 'b-', label='Mz')
            ax18.set_title('Control Torques M (Nm)')
            ax18.grid(True); ax18.legend()
            
            ax19 = fig.add_subplot(rows, 4, 19)
            for i in range(4):
                ax19.plot(time_array, motor_thrusts[:, i], label=f'Motor {i+1}')
            ax19.set_title('Individual Motor Thrusts (N)')
            ax19.grid(True); ax19.legend()
            
            ax20 = fig.add_subplot(rows, 4, 20, projection='3d')
            ax20.plot(position[:, 0], position[:, 1], position[:, 2], 'b-', label='Actual')
            ax20.plot(target_position[:, 0], target_position[:, 1], target_position[:, 2], 'r--', label='Ref')
            ax20.set_xlabel('X'); ax20.set_ylabel('Y'); ax20.set_zlabel('Z')
            ax20.set_title('UAV 3D Trajectory')
            ax20.legend()

            # --- Row 6: Delta Tracking (if applicable) ---
            if has_delta:
                delta_des = np.array(self.history['delta_des_pos'])
                delta_act = np.array(self.history['delta_actual_pos'])
                
                ax21 = fig.add_subplot(rows, 4, 21, projection='3d')
                ax21.plot(delta_des[:, 0], delta_des[:, 1], delta_des[:, 2], 'r--', label='Desired')
                ax21.plot(delta_act[:, 0], delta_act[:, 1], delta_act[:, 2], 'b-', label='Actual')
                ax21.set_xlabel('X'); ax21.set_ylabel('Y'); ax21.set_zlabel('Z')
                ax21.set_title('Delta 3D Trajectory')
                ax21.legend()
                
                ax22 = fig.add_subplot(rows, 4, 22)
                ax22.plot(time_array, delta_des[:, 0] - delta_act[:, 0], 'r-', label='Err X')
                ax22.set_title('Delta X Error (m)')
                ax22.grid(True); ax22.legend()
                
                ax23 = fig.add_subplot(rows, 4, 23)
                ax23.plot(time_array, delta_des[:, 1] - delta_act[:, 1], 'g-', label='Err Y')
                ax23.set_title('Delta Y Error (m)')
                ax23.grid(True); ax23.legend()
                
                ax24 = fig.add_subplot(rows, 4, 24)
                ax24.plot(time_array, delta_des[:, 2] - delta_act[:, 2], 'b-', label='Err Z')
                ax24.set_title('Delta Z Error (m)')
                ax24.grid(True); ax24.legend()

            plt.tight_layout()
            
            # Always save figure in headless mode
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error plotting results: {e}")

    # =========================================================================
    # Delta-Specific Methods
    # =========================================================================
    
    def reset(self):
        """
        Clears all recorded history data.
        Useful for resetting between simulation runs.
        """
        for key in self.history:
            self.history[key] = []
    
    def log_delta(self, time, des_pos, actual_pos, 
                  des_vel=None, actual_vel=None,
                  motor_angles=None, motor_vels=None,
                  ik_success=True, workspace_status=False):
        """
        Records a single data point for Delta robot simulation.
        Simplified interface compared to log() - only requires essential Delta data.
        
        Args:
            time: Current simulation time (float).
            des_pos: Desired end-effector position [x, y, z] (np.ndarray or list).
            actual_pos: Actual end-effector position [x, y, z] (np.ndarray or list).
            des_vel: Desired velocity [vx, vy, vz] (optional).
            actual_vel: Actual velocity [vx, vy, vz] (optional).
            motor_angles: Motor angles [θ1, θ2, θ3] in radians (optional).
            motor_vels: Motor velocities [ω1, ω2, ω3] in rad/s (optional).
            ik_success: Whether IK succeeded for this step (bool).
            workspace_status: Whether target was outside workspace (bool).
        """
        self.history['time'].append(time)
        
        # Position data
        self.history['delta_des_pos'].append(np.array(des_pos).copy())
        self.history['delta_actual_pos'].append(np.array(actual_pos).copy())
        
        # Velocity data (optional)
        if des_vel is not None:
            self.history['delta_des_vel'].append(np.array(des_vel).copy())
        else:
            self.history['delta_des_vel'].append(np.zeros(3))
        
        if actual_vel is not None:
            self.history['delta_actual_vel'].append(np.array(actual_vel).copy())
        else:
            self.history['delta_actual_vel'].append(np.zeros(3))
        
        # Motor data (optional)
        if motor_angles is not None:
            self.history['delta_motor_angles'].append(np.array(motor_angles).copy())
        else:
            self.history['delta_motor_angles'].append(np.zeros(3))
        
        if motor_vels is not None:
            self.history['delta_motor_velocities'].append(np.array(motor_vels).copy())
        else:
            self.history['delta_motor_velocities'].append(np.zeros(3))
        
        # Status flags
        self.history['delta_ik_success'].append(ik_success)
        self.history['delta_workspace_status'].append(workspace_status)
    
    def _compute_workspace_boundary(self, max_radius=0.12, z_min=-0.25, z_max=0.0, 
                                     theta_samples=36, z_samples=10):
        """
        Computes the Delta robot workspace boundary points for 3D visualization.
        
        The workspace is approximated as an inverted cone (frustum).
        
        Args:
            max_radius: Maximum horizontal radius at z_min.
            z_min: Minimum Z value (lowest position).
            z_max: Maximum Z value (at base level).
            theta_samples: Number of angular samples.
            z_samples: Number of Z level samples.
        
        Returns:
            dict with 'top', 'bottom', 'side' point arrays for plotting.
        """
        # Top circle (at z_max, radius=0)
        top_circle = []
        for theta in np.linspace(0, 2*np.pi, theta_samples):
            top_circle.append([0, 0, z_max])
        top_circle = np.array(top_circle)
        
        # Bottom circle (at z_min, radius=max_radius)
        bottom_circle = []
        for theta in np.linspace(0, 2*np.pi, theta_samples):
            x = max_radius * np.cos(theta)
            y = max_radius * np.sin(theta)
            bottom_circle.append([x, y, z_min])
        bottom_circle = np.array(bottom_circle)
        
        # Side surface (conical)
        side_surface = {'x': [], 'y': [], 'z': []}
        for z in np.linspace(z_min, z_max, z_samples):
            # Radius decreases linearly from max_radius at z_min to 0 at z_max
            r = max_radius * (z - z_max) / (z_min - z_max)
            for theta in np.linspace(0, 2*np.pi, theta_samples):
                side_surface['x'].append(r * np.cos(theta))
                side_surface['y'].append(r * np.sin(theta))
                side_surface['z'].append(z)
        
        return {
            'top': top_circle,
            'bottom': bottom_circle,
            'side': side_surface
        }
    
    def plot_delta_results(self, save_path='delta_simulation_results.png', 
                           show_workspace_boundary=True,
                           workspace_params=None):
        """
        Plots Delta robot simulation results with specialized layout.
        
        Layout (4 rows × 4 columns):
            Row 1: End-effector position X/Y/Z + position error norm
            Row 2: End-effector velocity Vx/Vy/Vz + velocity error norm
            Row 3: Motor angles θ1/θ2/θ3 + motor rate
            Row 4: Position error X/Y/Z + IK status
            Row 5: 3D trajectory with workspace boundary (optional)
        
        Args:
            save_path: Path to save the figure.
            show_workspace_boundary: Whether to show workspace boundary in 3D plot.
            workspace_params: Dict with 'max_radius', 'z_min', 'z_max' for boundary.
        """
        try:
            time_array = np.array(self.history['time'])
            if len(time_array) == 0:
                print("No data to plot.")
                return
            
            # Extract data
            des_pos = np.array(self.history['delta_des_pos'])
            actual_pos = np.array(self.history['delta_actual_pos'])
            des_vel = np.array(self.history['delta_des_vel'])
            actual_vel = np.array(self.history['delta_actual_vel'])
            motor_angles = np.array(self.history['delta_motor_angles'])
            motor_vels = np.array(self.history['delta_motor_velocities'])
            ik_success = np.array(self.history['delta_ik_success'])
            workspace_status = np.array(self.history['delta_workspace_status'])
            
            # Convert motor angles from radians to degrees
            motor_angles_deg = np.rad2deg(motor_angles)
            motor_vels_deg = np.rad2deg(motor_vels)
            
            # Calculate errors
            pos_error = des_pos - actual_pos
            pos_error_norm = np.linalg.norm(pos_error, axis=1)
            vel_error = des_vel - actual_vel
            vel_error_norm = np.linalg.norm(vel_error, axis=1)
            
            # Setup figure
            rows = 5
            fig = plt.figure(figsize=(20, 4 * rows))
            
            # --- Row 1: Position ---
            ax1 = fig.add_subplot(rows, 4, 1)
            ax1.plot(time_array, actual_pos[:, 0], 'r-', label='Act X')
            ax1.plot(time_array, des_pos[:, 0], 'r--', label='Des X')
            ax1.set_title('Position X (m)')
            ax1.grid(True); ax1.legend()
            
            ax2 = fig.add_subplot(rows, 4, 2)
            ax2.plot(time_array, actual_pos[:, 1], 'g-', label='Act Y')
            ax2.plot(time_array, des_pos[:, 1], 'g--', label='Des Y')
            ax2.set_title('Position Y (m)')
            ax2.grid(True); ax2.legend()
            
            ax3 = fig.add_subplot(rows, 4, 3)
            ax3.plot(time_array, actual_pos[:, 2], 'b-', label='Act Z')
            ax3.plot(time_array, des_pos[:, 2], 'b--', label='Des Z')
            ax3.set_title('Position Z (m)')
            ax3.grid(True); ax3.legend()
            
            ax4 = fig.add_subplot(rows, 4, 4)
            ax4.plot(time_array, pos_error_norm, 'k-', label='Error')
            ax4.set_title('Position Error Norm (m)')
            ax4.grid(True); ax4.legend()
            
            # --- Row 2: Velocity ---
            ax5 = fig.add_subplot(rows, 4, 5)
            ax5.plot(time_array, actual_vel[:, 0], 'r-', label='Act Vx')
            ax5.plot(time_array, des_vel[:, 0], 'r--', label='Des Vx')
            ax5.set_title('Velocity X (m/s)')
            ax5.grid(True); ax5.legend()
            
            ax6 = fig.add_subplot(rows, 4, 6)
            ax6.plot(time_array, actual_vel[:, 1], 'g-', label='Act Vy')
            ax6.plot(time_array, des_vel[:, 1], 'g--', label='Des Vy')
            ax6.set_title('Velocity Y (m/s)')
            ax6.grid(True); ax6.legend()
            
            ax7 = fig.add_subplot(rows, 4, 7)
            ax7.plot(time_array, actual_vel[:, 2], 'b-', label='Act Vz')
            ax7.plot(time_array, des_vel[:, 2], 'b--', label='Des Vz')
            ax7.set_title('Velocity Z (m/s)')
            ax7.grid(True); ax7.legend()
            
            ax8 = fig.add_subplot(rows, 4, 8)
            ax8.plot(time_array, vel_error_norm, 'k-', label='Error')
            ax8.set_title('Velocity Error Norm (m/s)')
            ax8.grid(True); ax8.legend()
            
            # --- Row 3: Motor Angles ---
            ax9 = fig.add_subplot(rows, 4, 9)
            ax9.plot(time_array, motor_angles_deg[:, 0], 'r-', label='θ1')
            ax9.set_title('Motor θ1 (deg)')
            ax9.grid(True); ax9.legend()
            
            ax10 = fig.add_subplot(rows, 4, 10)
            ax10.plot(time_array, motor_angles_deg[:, 1], 'g-', label='θ2')
            ax10.set_title('Motor θ2 (deg)')
            ax10.grid(True); ax10.legend()
            
            ax11 = fig.add_subplot(rows, 4, 11)
            ax11.plot(time_array, motor_angles_deg[:, 2], 'b-', label='θ3')
            ax11.set_title('Motor θ3 (deg)')
            ax11.grid(True); ax11.legend()
            
            ax12 = fig.add_subplot(rows, 4, 12)
            total_motor_rate = np.sum(np.abs(motor_vels_deg), axis=1)
            ax12.plot(time_array, total_motor_rate, 'k-', label='Sum |ω|')
            ax12.set_title('Total Motor Rate (deg/s)')
            ax12.grid(True); ax12.legend()
            
            # --- Row 4: Position Error Details + IK Status ---
            ax13 = fig.add_subplot(rows, 4, 13)
            ax13.plot(time_array, pos_error[:, 0], 'r-', label='Err X')
            ax13.set_title('X Error (m)')
            ax13.grid(True); ax13.legend()
            
            ax14 = fig.add_subplot(rows, 4, 14)
            ax14.plot(time_array, pos_error[:, 1], 'g-', label='Err Y')
            ax14.set_title('Y Error (m)')
            ax14.grid(True); ax14.legend()
            
            ax15 = fig.add_subplot(rows, 4, 15)
            ax15.plot(time_array, pos_error[:, 2], 'b-', label='Err Z')
            ax15.set_title('Z Error (m)')
            ax15.grid(True); ax15.legend()
            
            ax16 = fig.add_subplot(rows, 4, 16)
            # IK status as color-coded bars
            ik_numeric = ik_success.astype(float)
            ax16.fill_between(time_array, 0, 1, where=ik_success, color='green', alpha=0.5, label='IK OK')
            ax16.fill_between(time_array, 0, 1, where=~ik_success, color='red', alpha=0.5, label='IK Fail')
            # Mark workspace boundary violations
            if np.any(workspace_status):
                ax16.scatter(time_array[workspace_status], 
                            np.ones(np.sum(workspace_status)) * 0.5,
                            c='orange', s=20, marker='x', label='Outside WS')
            ax16.set_ylim(0, 1.2)
            ax16.set_title('IK Status')
            ax16.legend(loc='upper right')
            ax16.set_yticks([])
            
            # --- Row 5: 3D Trajectory with Workspace Boundary ---
            ax17 = fig.add_subplot(rows, 4, 17, projection='3d')
            
            # Plot workspace boundary if requested
            if show_workspace_boundary:
                if workspace_params is None:
                    workspace_params = {
                        'max_radius': 0.12,
                        'z_min': -0.25,
                        'z_max': 0.0
                    }
                
                boundary = self._compute_workspace_boundary(**workspace_params)
                
                # Plot bottom circle
                ax17.plot(boundary['bottom'][:, 0], boundary['bottom'][:, 1], 
                         boundary['bottom'][:, 2], 'k--', alpha=0.5, label='Workspace')
                
                # Plot side lines (connecting top to bottom)
                n_lines = 8
                for i in range(n_lines):
                    idx = int(i * len(boundary['bottom']) / n_lines)
                    ax17.plot([0, boundary['bottom'][idx, 0]], 
                             [0, boundary['bottom'][idx, 1]], 
                             [workspace_params['z_max'], workspace_params['z_min']], 
                             'k--', alpha=0.3)
            
            # Plot trajectories
            ax17.plot(des_pos[:, 0], des_pos[:, 1], des_pos[:, 2], 
                     'r--', label='Desired', linewidth=1.5)
            ax17.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], 
                     'b-', label='Actual', linewidth=1.5)
            
            # Mark start and end points
            ax17.scatter(*actual_pos[0], c='green', s=100, marker='o', label='Start')
            ax17.scatter(*actual_pos[-1], c='blue', s=100, marker='s', label='End')
            
            ax17.set_xlabel('X (m)')
            ax17.set_ylabel('Y (m)')
            ax17.set_zlabel('Z (m)')
            ax17.set_title('End-Effector 3D Trajectory')
            ax17.legend(loc='upper left', fontsize=8)
            
            # --- Remaining subplots for additional analysis ---
            # XY projection
            ax18 = fig.add_subplot(rows, 4, 18)
            ax18.plot(des_pos[:, 0], des_pos[:, 1], 'r--', label='Desired')
            ax18.plot(actual_pos[:, 0], actual_pos[:, 1], 'b-', label='Actual')
            ax18.set_xlabel('X (m)')
            ax18.set_ylabel('Y (m)')
            ax18.set_title('XY Projection')
            ax18.grid(True); ax18.legend()
            ax18.set_aspect('equal')
            
            # XZ projection
            ax19 = fig.add_subplot(rows, 4, 19)
            ax19.plot(des_pos[:, 0], des_pos[:, 2], 'r--', label='Desired')
            ax19.plot(actual_pos[:, 0], actual_pos[:, 2], 'b-', label='Actual')
            ax19.set_xlabel('X (m)')
            ax19.set_ylabel('Z (m)')
            ax19.set_title('XZ Projection')
            ax19.grid(True); ax19.legend()
            
            # YZ projection
            ax20 = fig.add_subplot(rows, 4, 20)
            ax20.plot(des_pos[:, 1], des_pos[:, 2], 'r--', label='Desired')
            ax20.plot(actual_pos[:, 1], actual_pos[:, 2], 'b-', label='Actual')
            ax20.set_xlabel('Y (m)')
            ax20.set_ylabel('Z (m)')
            ax20.set_title('YZ Projection')
            ax20.grid(True); ax20.legend()
            
            plt.tight_layout()
            
            # Save figure
            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
            
            # Print summary statistics
            self._print_delta_summary(pos_error_norm, ik_success, workspace_status)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error plotting Delta results: {e}")
    
    def _print_delta_summary(self, pos_error_norm, ik_success, workspace_status):
        """
        Prints summary statistics for Delta simulation.
        """
        print("\n" + "="*50)
        print("Delta Simulation Summary")
        print("="*50)
        print(f"Total steps: {len(pos_error_norm)}")
        print(f"Position error - Mean: {np.mean(pos_error_norm):.6f} m")
        print(f"Position error - Max:  {np.max(pos_error_norm):.6f} m")
        print(f"Position error - Std:  {np.std(pos_error_norm):.6f} m")
        print(f"IK success rate: {np.sum(ik_success)/len(ik_success)*100:.1f}%")
        if np.any(workspace_status):
            print(f"Workspace boundary violations: {np.sum(workspace_status)} steps")
        print("="*50 + "\n")
    
    # =========================================================================
    # Multi-Ball Interception Methods
    # =========================================================================
    
    def log_interception(self, ball_id: int, ball_trajectory: list, 
                         ee_trajectory: list, catch_time: float,
                         is_caught: bool, intercept_point: np.ndarray):
        """
        Log data for a single ball interception attempt.
        
        Args:
            ball_id: Ball identifier (0-99).
            ball_trajectory: List of ball positions [(x,y,z), ...].
            ee_trajectory: List of end-effector positions [(x,y,z), ...].
            catch_time: Time when catch occurred (or -1 if missed).
            is_caught: Whether ball was successfully caught.
            intercept_point: Planned intercept point.
        """
        if not hasattr(self, 'interception_data'):
            self.interception_data = {
                'ball_ids': [],
                'ball_trajectories': [],
                'ee_trajectories': [],
                'catch_times': [],
                'is_caught': [],
                'intercept_points': []
            }
        
        self.interception_data['ball_ids'].append(ball_id)
        self.interception_data['ball_trajectories'].append(ball_trajectory)
        self.interception_data['ee_trajectories'].append(ee_trajectory)
        self.interception_data['catch_times'].append(catch_time)
        self.interception_data['is_caught'].append(is_caught)
        self.interception_data['intercept_points'].append(intercept_point)
    
    def plot_multi_interception_results(self, save_path='multi_interception_results.png'):
        """
        Create comprehensive visualization for multi-ball interception simulation.
        
        Includes:
        1. 3D trajectories of all balls and end-effector paths
        2. Catch success rate statistics
        3. Catch time distribution
        4. Position error analysis
        5. XY and XZ projections
        """
        if not hasattr(self, 'interception_data'):
            print("No interception data to plot.")
            return
        
        data = self.interception_data
        n_balls = len(data['ball_ids'])
        n_caught = sum(data['is_caught'])
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # --- Plot 1: 3D Trajectories (all balls + EE) ---
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        # Color map for caught/missed
        caught_color = 'green'
        missed_color = 'red'
        ee_color = 'blue'
        
        for i, (ball_traj, ee_traj, caught) in enumerate(
                zip(data['ball_trajectories'], data['ee_trajectories'], data['is_caught'])):
            
            ball_traj = np.array(ball_traj)
            ee_traj = np.array(ee_traj)
            
            color = caught_color if caught else missed_color
            alpha = 0.7 if caught else 0.3
            
            # Plot ball trajectory (skip if too short)
            if len(ball_traj) >= 2 and ball_traj.ndim == 2 and ball_traj.shape[1] >= 3:
                ax1.plot(ball_traj[:, 0], ball_traj[:, 1], ball_traj[:, 2],
                        color=color, alpha=alpha, linewidth=0.8)
            
            # Plot EE trajectory (thinner line, skip if too short)
            if len(ee_traj) >= 2 and ee_traj.ndim == 2 and ee_traj.shape[1] >= 3:
                ax1.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2],
                        color=ee_color, alpha=0.5, linewidth=0.5)
        
        # Add workspace boundary (approximate)
        theta = np.linspace(0, 2*np.pi, 50)
        for z in [-0.05, -0.20]:
            r = 0.10 * (z + 0.05) / (-0.20 + 0.05) if z < -0.05 else 0.0
            r = abs(r) if r != 0 else 0.10
            ax1.plot(r * np.cos(theta), r * np.sin(theta), 
                    np.full_like(theta, z + 0.5), 'k--', alpha=0.3)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'3D Trajectories\n(Green=Caught, Red=Missed, Blue=EE Path)\n{n_caught}/{n_balls} Caught')
        
        # --- Plot 2: Success Rate Pie Chart ---
        ax2 = fig.add_subplot(2, 3, 2)
        
        labels = ['Caught', 'Missed']
        sizes = [n_caught, n_balls - n_caught]
        colors = [caught_color, missed_color]
        explode = (0.05, 0)
        
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax2.set_title(f'Catch Success Rate: {n_caught}/{n_balls} ({100*n_caught/n_balls:.1f}%)')
        
        # --- Plot 3: Catch Time Distribution ---
        ax3 = fig.add_subplot(2, 3, 3)
        
        catch_times = [t for t, caught in zip(data['catch_times'], data['is_caught']) if caught and t > 0]
        if catch_times:
            ax3.hist(catch_times, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(catch_times), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(catch_times):.3f}s')
            ax3.set_xlabel('Catch Time (s)')
            ax3.set_ylabel('Count')
            ax3.set_title('Catch Time Distribution')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No catches recorded', ha='center', va='center')
            ax3.set_title('Catch Time Distribution (No Data)')
        
        # --- Plot 4: XY Projection ---
        ax4 = fig.add_subplot(2, 3, 4)
        
        for i, (ball_traj, ee_traj, caught) in enumerate(
                zip(data['ball_trajectories'], data['ee_trajectories'], data['is_caught'])):
            
            ball_traj = np.array(ball_traj)
            ee_traj = np.array(ee_traj)
            
            color = caught_color if caught else missed_color
            alpha = 0.7 if caught else 0.3
            
            if len(ball_traj) >= 2 and ball_traj.ndim == 2 and ball_traj.shape[1] >= 2:
                ax4.plot(ball_traj[:, 0], ball_traj[:, 1], color=color, alpha=alpha, linewidth=0.8)
            if len(ee_traj) >= 2 and ee_traj.ndim == 2 and ee_traj.shape[1] >= 2:
                ax4.plot(ee_traj[:, 0], ee_traj[:, 1], color=ee_color, alpha=0.3, linewidth=0.5)
        
        # Draw workspace circle
        workspace_circle = plt.Circle((0, 0), 0.10, fill=False, linestyle='--', color='black')
        ax4.add_patch(workspace_circle)
        
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_title('Top View (XY Projection)')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
        # --- Plot 5: XZ Projection ---
        ax5 = fig.add_subplot(2, 3, 5)
        
        for i, (ball_traj, ee_traj, caught) in enumerate(
                zip(data['ball_trajectories'], data['ee_trajectories'], data['is_caught'])):
            
            ball_traj = np.array(ball_traj)
            ee_traj = np.array(ee_traj)
            
            color = caught_color if caught else missed_color
            alpha = 0.7 if caught else 0.3
            
            if len(ball_traj) >= 2 and ball_traj.ndim == 2 and ball_traj.shape[1] >= 3:
                ax5.plot(ball_traj[:, 0], ball_traj[:, 2], color=color, alpha=alpha, linewidth=0.8)
            if len(ee_traj) >= 2 and ee_traj.ndim == 2 and ee_traj.shape[1] >= 3:
                ax5.plot(ee_traj[:, 0], ee_traj[:, 2], color=ee_color, alpha=0.3, linewidth=0.5)
        
        # Draw workspace region
        ax5.axhline(y=0.45, color='black', linestyle='--', alpha=0.5, label='Workspace Z max')
        ax5.axhline(y=0.30, color='black', linestyle='--', alpha=0.5, label='Workspace Z min')
        
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Z (m)')
        ax5.set_title('Side View (XZ Projection)')
        ax5.grid(True, alpha=0.3)
        
        # --- Plot 6: Ball-by-Ball Results ---
        ax6 = fig.add_subplot(2, 3, 6)
        
        ball_ids = data['ball_ids']
        results = [1 if caught else 0 for caught in data['is_caught']]
        
        colors_bar = [caught_color if r else missed_color for r in results]
        ax6.bar(ball_ids, results, color=colors_bar, alpha=0.7)
        
        ax6.set_xlabel('Ball ID')
        ax6.set_ylabel('Caught (1) / Missed (0)')
        ax6.set_title(f'Individual Ball Results\nSuccess Rate: {100*n_caught/n_balls:.1f}%')
        ax6.set_ylim(0, 1.2)
        
        # Add statistics text
        stats_text = f"""
Statistics:
Total Balls: {n_balls}
Caught: {n_caught}
Missed: {n_balls - n_caught}
Success Rate: {100*n_caught/n_balls:.1f}%
"""
        if catch_times:
            stats_text += f"Mean Catch Time: {np.mean(catch_times):.3f}s\n"
            stats_text += f"Std Catch Time: {np.std(catch_times):.3f}s"
        
        fig.text(0.98, 0.02, stats_text, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Multi-interception plot saved to {save_path}")
        
        plt.close()
    
    def plot_single_interception_3d(self, ball_id: int, save_path=None):
        """
        Plot 3D trajectory for a single ball interception.
        
        Args:
            ball_id: The ball ID to plot.
            save_path: Path to save the figure.
        """
        if not hasattr(self, 'interception_data'):
            print("No interception data available.")
            return
        
        idx = self.interception_data['ball_ids'].index(ball_id)
        
        ball_traj = np.array(self.interception_data['ball_trajectories'][idx])
        ee_traj = np.array(self.interception_data['ee_trajectories'][idx])
        caught = self.interception_data['is_caught'][idx]
        intercept_point = self.interception_data['intercept_points'][idx]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot ball trajectory (with dimension check)
        if len(ball_traj) >= 2 and ball_traj.ndim == 2 and ball_traj.shape[1] >= 3:
            ax.plot(ball_traj[:, 0], ball_traj[:, 1], ball_traj[:, 2],
                   'r-', linewidth=2, label='Ball Trajectory')
            ax.scatter(ball_traj[0, 0], ball_traj[0, 1], ball_traj[0, 2],
                      c='red', s=100, marker='o', label='Ball Start')
        
        # Plot EE trajectory (with dimension check)
        if len(ee_traj) >= 2 and ee_traj.ndim == 2 and ee_traj.shape[1] >= 3:
            ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2],
                   'b-', linewidth=2, label='EE Trajectory')
            ax.scatter(ee_traj[0, 0], ee_traj[0, 1], ee_traj[0, 2],
                      c='blue', s=100, marker='s', label='EE Start')
        
        # Plot intercept point
        if intercept_point is not None and np.any(intercept_point):
            ax.scatter(intercept_point[0], intercept_point[1], intercept_point[2],
                      c='green', s=150, marker='*', label='Intercept Point')
        
        # Mark catch point if caught
        if caught and len(ball_traj) >= 1 and ball_traj.ndim == 2:
            ax.scatter(ball_traj[-1, 0], ball_traj[-1, 1], ball_traj[-1, 2],
                      c='lime', s=200, marker='*', label='CATCH!')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Ball {ball_id} Interception {"(CAUGHT)" if caught else "(MISSED)"}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Single interception plot saved to {save_path}")
        
        plt.close()
    
    def get_interception_summary(self) -> dict:
        """
        Get summary statistics for multi-ball interception.
        
        Returns:
            Dict with summary statistics.
        """
        if not hasattr(self, 'interception_data'):
            return {}
        
        data = self.interception_data
        n_balls = len(data['ball_ids'])
        n_caught = sum(data['is_caught'])
        catch_times = [t for t, caught in zip(data['catch_times'], data['is_caught']) if caught and t > 0]
        
        return {
            'total_balls': n_balls,
            'caught': n_caught,
            'missed': n_balls - n_caught,
            'success_rate': n_caught / n_balls if n_balls > 0 else 0,
            'mean_catch_time': np.mean(catch_times) if catch_times else 0,
            'std_catch_time': np.std(catch_times) if catch_times else 0
        }
