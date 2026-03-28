"""
Delta robot model wrapper for MuJoCo interaction.
Standalone version for Delta-only simulation.
"""
import mujoco
import numpy as np


class DeltaModel:
    """
    Interface for the Delta robot in MuJoCo simulation.
    Handles sensor reading and actuator writing for Delta arm.
    """
    
    def __init__(self, model, data, base_name="Delta_base_body", 
                 end_platform_name="end_platform"):
        """
        Args:
            model: MuJoCo model object.
            data: MuJoCo data object.
            base_name: Name of the base body in XML.
            end_platform_name: Name of the end platform body in XML.
        """
        self.model = model
        self.data = data
        self.base_name = base_name
        self.end_platform_name = end_platform_name
        
        # Get body IDs
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_name)
        if self.base_id == -1:
            raise ValueError(f"Body '{base_name}' not found in model.")
        
        self.end_platform_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_platform_name)
        if self.end_platform_id == -1:
            raise ValueError(f"Body '{end_platform_name}' not found in model.")
        
        # Get sensor IDs
        self.ee_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_pos")
        self.ee_lin_vel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_lin_vel")
        
        if self.ee_sensor_id == -1:
            print("Warning: 'ee_pos' sensor not found. Position reading will return zeros.")
        if self.ee_lin_vel_sensor_id == -1:
            print("Warning: 'ee_lin_vel' sensor not found. Velocity reading will return zeros.")
        
        # Get motor actuator IDs
        self.motor_names = ['armmotor1', 'armmotor2', 'armmotor3']
        self.motor_ids = []
        for name in self.motor_names:
            motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if motor_id == -1:
                raise ValueError(f"Actuator '{name}' not found in model.")
            self.motor_ids.append(motor_id)
        
        # Workspace parameters (from DeltaKinematics)
        self.workspace_max_radius = 0.12  # m
        self.workspace_z_min = -0.25      # m (relative to base)
        self.workspace_z_max = 0.0        # m (relative to base)
    
    def get_ee_sensor_pos(self):
        """
        Returns the position of the end-effector relative to base.
        Reads from 'ee_pos' sensor.
        
        Returns:
            np.ndarray: Position [x, y, z] relative to base (3,).
        """
        if self.ee_sensor_id != -1:
            adr = self.model.sensor_adr[self.ee_sensor_id]
            dim = self.model.sensor_dim[self.ee_sensor_id]
            return self.data.sensordata[adr:adr+dim].copy()
        return np.zeros(3)
    
    def get_ee_sensor_lin_vel(self):
        """
        Returns the linear velocity of the end-effector relative to base.
        Reads from 'ee_lin_vel' sensor.
        
        Returns:
            np.ndarray: Velocity [vx, vy, vz] relative to base (3,).
        """
        if self.ee_lin_vel_sensor_id != -1:
            adr = self.model.sensor_adr[self.ee_lin_vel_sensor_id]
            dim = self.model.sensor_dim[self.ee_lin_vel_sensor_id]
            return self.data.sensordata[adr:adr+dim].copy()
        return np.zeros(3)
    
    def get_ee_world_pos(self):
        """
        Returns the position of the end-effector in world frame.
        
        Returns:
            np.ndarray: Position [x, y, z] in world frame (3,).
        """
        return self.data.body(self.end_platform_id).xpos.copy()
    
    def get_base_world_pos(self):
        """
        Returns the position of the base in world frame.
        
        Returns:
            np.ndarray: Position [x, y, z] in world frame (3,).
        """
        return self.data.body(self.base_id).xpos.copy()
    
    def get_motor_angles(self):
        """
        Returns current motor angles from actuators.
        
        Returns:
            np.ndarray: Motor angles [θ1, θ2, θ3] in radians (3,).
        """
        angles = np.array([self.data.actuator(id).ctrl[0] for id in self.motor_ids])
        return angles
    
    def get_motor_velocities(self):
        """
        Returns current motor velocities.
        
        Returns:
            np.ndarray: Motor velocities [ω1, ω2, ω3] in rad/s (3,).
        """
        vels = np.array([self.data.actuator_velocity[id] for id in self.motor_ids])
        return vels
    
    def set_delta_motor_positions(self, positions):
        """
        Sets the position of the Delta robot motors.
        
        Args:
            positions: [p1, p2, p3] angular positions (rad) for the 3 motors.
        """
        positions = np.array(positions)
        for i, motor_id in enumerate(self.motor_ids):
            self.data.actuator(motor_id).ctrl[0] = positions[i]
    
    def set_delta_motor_velocities(self, velocities):
        """
        Sets the velocity of the Delta robot motors (for velocity control mode).
        
        Args:
            velocities: [v1, v2, v3] angular velocities (rad/s) for the 3 motors.
        """
        velocities = np.array(velocities)
        # Note: This requires velocity actuators in XML
        # For now, this is a placeholder
        for i, name in enumerate(self.motor_names):
            vel_actuator_name = f"{name}_vel"
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, vel_actuator_name)
            if actuator_id != -1:
                self.data.actuator(actuator_id).ctrl[0] = velocities[i]
    
    def print_state(self):
        """
        Prints the current state of the Delta robot.
        """
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        
        ee_pos = self.get_ee_sensor_pos()
        ee_vel = self.get_ee_sensor_lin_vel()
        motor_angles = self.get_motor_angles()
        motor_angles_deg = np.rad2deg(motor_angles)
        
        print("=== Delta Robot State ===")
        print(f"End-Effector Position (rel): {ee_pos}")
        print(f"End-Effector Velocity (rel): {ee_vel}")
        print(f"Motor Angles (deg): {motor_angles_deg}")
        print(f"Base Position (world): {self.get_base_world_pos()}")
        print("=========================")
