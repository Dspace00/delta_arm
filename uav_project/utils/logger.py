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
    """
    def __init__(self):
        self.history = {
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
            'delta_des_pos': [],
            'delta_actual_pos': []
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
