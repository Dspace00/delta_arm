"""
UAV + delta Model wrapper for MuJoCo interaction.
"""
import os
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

class UAVModel:
    """
    Interface for the UAV in the MuJoCo simulation.
    Handles sensor reading and actuator writing.
    """
    def __init__(self, model, data, body_name="UAV_body", base_name="Delta_base_body", end_platform_name="end_platform"):
        """
        Args:
            model: MuJoCo model object.
            data: MuJoCo data object.
            body_name: Name of the UAV body in XML.
            base_name: Name of the base body in XML.
            end_platform_name: Name of the end platform body in XML.
        """
        self.model = model
        self.data = data
        self.body_name = body_name
        self.base_name = base_name
        self.end_platform_name = end_platform_name
        
        # Get body ID
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model.")
        
        # Check if Delta arm exists
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_name)
        self.end_platform_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_platform_name)
        
        if self.base_id != -1 and self.end_platform_id != -1:
            self.has_delta = True
            self.ee_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_pos")
            self.ee_lin_vel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_lin_vel")
        else:
            self.has_delta = False
            self.ee_sensor_id = -1
            self.ee_lin_vel_sensor_id = -1

        # --- Dynamic Motor Parameter Update ---
        from ..config import CT, CM
        for i in range(4):
            actuator_name = f'rotor{i}'
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id != -1:
                # Update the gear ratio for thrust (index 2) and torque (index 5)
                # gear format: [0, 0, CT, 0, 0, +/-CM]
                # Rotor 0, 2: CCW -> -CM
                # Rotor 1, 3: CW  -> +CM
                self.model.actuator_gear[actuator_id, 2] = CT
                self.model.actuator_gear[actuator_id, 5] = -CM if i in [0, 2] else CM



    def get_mass(self):
        """
        Returns the subtree mass of the UAV body.
        """
        return self.model.body_subtreemass[self.body_id]

    def get_inertia(self):
        """
        Returns the diagonal inertia of the UAV body.
        """
        return self.model.body_inertia[self.body_id]

    def get_uav_state(self):
        """
        Returns the current state of the UAV.
        
        Returns:
            UAV_position: [x, y, z] (World frame)
            UAV_velocity: [vx, vy, vz] (World frame)
            UAV_attitude: [w, x, y, z] (Quaternion, World frame)
            UAV_angle_rate: [p, q, r] (Body frame)
        """
        # Position [x, y, z]
        UAV_position = self.data.body(self.body_id).xpos.copy()
        
        # Quaternion [w, x, y, z]
        UAV_quaternion = self.data.body(self.body_id).xquat.copy()
        
        # Rotation Matrix
        UAV_att_mat = self.data.body(self.body_id).xmat.copy().reshape(3, 3)
        
        # Linear Velocity [vx, vy, vz] (World frame)
        # cvel is [rot_vel, lin_vel] in world frame? No, cvel is com velocity (6D).
        # indices 3:6 are linear velocity in world frame.
        UAV_velocity = self.data.body(self.body_id).cvel[3:6].copy()
        
        # Angular Velocity [wx, wy, wz] (World frame)
        UAV_angle_rate_world = self.data.body(self.body_id).cvel[0:3].copy()
        
        # Convert Angular Velocity to Body Frame
        # w_body = R_body_to_world^T * w_world
        UAV_angle_rate = UAV_att_mat.T @ UAV_angle_rate_world
        
        return UAV_position, UAV_velocity, UAV_quaternion, UAV_angle_rate
    
    def get_delta_state(self):
        """
        Returns the current state of the Delta model.
        
        Returns:
            End_platform_position: [x, y, z] (World frame)
            End_platform_velocity: [vx, vy, vz] (World frame)

        """
        if not getattr(self, 'has_delta', False):
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        # Position [x, y, z] (World frame)
        End_platform_position = self.data.body(self.end_platform_id).xpos.copy()
        
        # Linear Velocity [vx, vy, vz] (World frame)
        End_platform_velocity = self.data.body(self.end_platform_id).cvel[3:6].copy()
    
        # Base position [x, y, z] (World frame)
        Base_position = self.data.body(self.base_id).xpos.copy()

        # Base velocity [vx, vy, vz] (World frame)
        Base_velocity = self.data.body(self.base_id).cvel[3:6].copy()

        # Relative position [x, y, z] (World frame)
        Relative_position = End_platform_position - Base_position
        
        # Relative velocity [vx, vy, vz] (World frame)
        Relative_velocity = End_platform_velocity - Base_velocity
        
        return End_platform_position, End_platform_velocity, Base_position, Base_velocity, Relative_position, Relative_velocity

    def print_uav_state(self):
        """
        Prints the current state of the UAV.
        """
        UAV_pos, UAV_vel, UAV_att, UAV_ang_rate = self.get_uav_state()
        
        os.system('clear')
        # 设置numpy打印选项
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        print(f"UAV Position: {UAV_pos}")
        print(f"UAV Velocity: {UAV_vel}")
        print(f"UAV Attitude (Quaternion): {UAV_att}")
        print(f"UAV Angle Rate: {UAV_ang_rate}")

        if getattr(self, 'has_delta', False):
            End_platform_pos, End_platform_vel, Base_pos, Base_vel, Relative_pos, Relative_vel = self.get_delta_state()
            relative_pos = self.get_ee_sensor_pos()
            relative_vel = self.get_ee_sensor_lin_vel()
            
            print(f"End Platform Position: {End_platform_pos}")
            print(f"End Platform Velocity: {End_platform_vel}")
            print(f"Base Position: {Base_pos}")
            print(f"Base Velocity: {Base_vel}")
            print(f"End Platform Sensor Position: {relative_pos}")
            print(f"End Platform Sensor Linear Velocity: {relative_vel}")

    def get_imu_data(self):
        """
        Returns sensor data (simulated IMU).
        """
        # Assuming standard sensor layout (accel, gyro, etc.)
        # This depends on the XML definition.
        # For now, just return a copy of sensordata
        return self.data.sensordata.copy()

    def get_ee_sensor_pos(self):
        """
        Returns the position of the end platform sensor.
        Reads from 'ee_pos' sensor if available.
        """
        if self.ee_sensor_id != -1:
            adr = self.model.sensor_adr[self.ee_sensor_id]
            dim = self.model.sensor_dim[self.ee_sensor_id]
            return self.data.sensordata[adr:adr+dim].copy() 
        return np.zeros(3)

    def get_ee_sensor_lin_vel(self):
        """
        Returns the linear velocity of the end platform sensor.
        Reads from 'ee_lin_vel' sensor if available.
        """
        if self.ee_lin_vel_sensor_id != -1:
            adr = self.model.sensor_adr[self.ee_lin_vel_sensor_id]
            dim = self.model.sensor_dim[self.ee_lin_vel_sensor_id]
            return self.data.sensordata[adr:adr+dim].copy() 
        return np.zeros(3) 
        

    def apply_simplified_controls(self, force_vector, torque_vector):
        """
        Applies simplified control inputs (Force vector + Torque vector) directly to actuators.
        
        Args:
            force_vector: [Fx, Fy, Fz] in BODY frame.
            torque_vector: [Mx, My, Mz] in BODY frame.
        """
        # Transform force from Body to World frame? 
        # Wait, the XML defines actuators:
        # <motor name="forcex" gear="1 0 0 0 0 0" site="UAV_center"/> 
        # Site "UAV_center" rotates with the body.
        # So "forcex" applies force along the site's X axis (which is Body X).
        # So we should pass BODY frame forces directly if we use the actuators.
        
        # However, the original code did this:
        # att_mat = self.data.body("UAV_body").xmat.copy().reshape(3,3)
        # force_body = att_mat.T @ self.controller_mix[0:3]
        # This implies controller_mix[0:3] was in WORLD frame (calculated as total_thrust * z_axis_world),
        # and then projected to BODY frame.
        
        # Let's assume the controller outputs WORLD frame force (vector) or BODY frame force?
        # Standard cascade controller outputs a Thrust magnitude along Z-body, 
        # OR a Force vector in World frame (which then needs to be achieved by attitude).
        
        # In the original code:
        # total_thrust = mass * acc_prop (Vector in World Frame, roughly)
        # self.controller_mix[0:3] = total_thrust
        # Then:
        # force_body = att_mat.T @ self.controller_mix[0:3]
        # So yes, the input to this function should be WORLD frame force if we want to reproduce logic,
        # OR we change the logic to accept Body frame force.
        
        # To be safe and clean: Let's accept FORCE VECTOR in WORLD FRAME, and TORQUE in BODY FRAME.
        # Because Torques are always applied in body frame for quadrotors.
        pass

    def set_actuators(self, force_body, torque_body):
        """
        Low-level actuator setting.
        
        Args:
            force_body: [Fx, Fy, Fz] in Body frame.
            torque_body: [Mx, My, Mz] in Body frame.
        """
        # Safe assignment checking if actuators exist
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'forcex') != -1:
            self.data.actuator('forcex').ctrl[0] = force_body[0]
            self.data.actuator('forcey').ctrl[0] = force_body[1]
            self.data.actuator('forcez').ctrl[0] = force_body[2]
            self.data.actuator('Mx').ctrl[0] = torque_body[0]
            self.data.actuator('My').ctrl[0] = torque_body[1]
            self.data.actuator('Mz').ctrl[0] = torque_body[2]

    def set_delta_motor_velocities(self, velocities):
        """
        Sets the velocity of the Delta robot motors.
        
        Args:
            velocities: [v1, v2, v3] angular velocities for the 3 motors.
        """
        # Check if velocity actuators exist before setting
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor1_vel') != -1:
            self.data.actuator('motor1_vel').ctrl[0] = velocities[0]
            self.data.actuator('motor2_vel').ctrl[0] = velocities[1]
            self.data.actuator('motor3_vel').ctrl[0] = velocities[2]

    def set_delta_motor_positions(self, positions):
        """
        Sets the position of the Delta robot motors.
        
        Args:
            positions: [p1, p2, p3] angular positions (rad) for the 3 motors.
        """
        # Assuming actuator names are armmotor1, armmotor2, armmotor3
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'armmotor1') != -1:
            self.data.actuator('armmotor1').ctrl[0] = positions[0]
            self.data.actuator('armmotor2').ctrl[0] = positions[1]
            self.data.actuator('armmotor3').ctrl[0] = positions[2]

    def set_motor_speeds(self, motor_speeds):
        """
        Sets the motor speeds for rotor actuators.
        
        Args:
            motor_speeds: List/Array of 4 motor speeds.
        """
        for i in range(4):
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'rotor{i}') != -1:
                self.data.actuator(f'rotor{i}').ctrl[0] = motor_speeds[i]
