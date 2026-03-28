"""
Extended Delta model with ball sensor support.
For ball interception simulation.

Data collection point: 10cm above end-effector platform center
(as per user requirement: vertical line through platform center, 10cm above center).
"""
import mujoco
import numpy as np
from uav_project.models.delta_model import DeltaModel


class DeltaBallModel(DeltaModel):
    """
    Extended Delta model with ball position/velocity sensors.
    
    Sensor offset: All end-effector position data is measured at a point
    10cm above the platform center (perpendicular to platform, through center).
    """
    
    def __init__(self, model, data, base_name="Delta_base_body", 
                 end_platform_name="end_platform", ball_name="ball",
                 sensor_offset_z: float = 0.10):
        """
        Args:
            model: MuJoCo model object.
            data: MuJoCo data object.
            base_name: Name of the base body in XML.
            end_platform_name: Name of the end platform body in XML.
            ball_name: Name of the ball body in XML.
            sensor_offset_z: Z offset for sensor measurement point (default 10cm).
        """
        super().__init__(model, data, base_name, end_platform_name)
        
        # Sensor offset: measurement point is above platform center
        # This is in the platform's local frame (Z is perpendicular to platform)
        self.sensor_offset_z = sensor_offset_z
        
        # Get ball body ID
        self.ball_name = ball_name
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ball_name)
        if self.ball_id == -1:
            raise ValueError(f"Body '{ball_name}' not found in model.")
        
        # Get ball sensor IDs
        self.ball_pos_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ball_pos")
        self.ball_lin_vel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ball_lin_vel")
        
        if self.ball_pos_sensor_id == -1:
            print("Warning: 'ball_pos' sensor not found.")
        if self.ball_lin_vel_sensor_id == -1:
            print("Warning: 'ball_lin_vel' sensor not found.")
        
        print(f"DeltaBallModel initialized with sensor offset: {self.sensor_offset_z*100:.1f}cm above platform")
    
    def get_ee_sensor_pos(self) -> np.ndarray:
        """
        Returns end-effector sensor position with offset.
        
        The measurement point is on the vertical line through platform center,
        at a distance of sensor_offset_z above the platform.
        
        Returns:
            np.ndarray: Position [x, y, z] relative to base (3,).
        """
        # Get platform position (relative to base)
        # Use end_platform_id from parent class DeltaModel
        platform_pos = self.data.body(self.end_platform_id).xpos.copy()
        base_pos = self.data.body(self.base_id).xpos.copy()
        
        # Position relative to base
        pos_relative = platform_pos - base_pos
        
        # Add sensor offset in Z direction
        # "Above" means less negative (closer to base which is at z=0)
        pos_relative[2] += self.sensor_offset_z
        
        return pos_relative
    
    def get_ee_sensor_lin_vel(self) -> np.ndarray:
        """
        Returns end-effector linear velocity.
        
        The velocity at the sensor point is approximately the same as 
        platform velocity for small rotations.
        
        Returns:
            np.ndarray: Velocity [vx, vy, vz] (3,).
        """
        # Platform velocity (use end_platform_id from parent class)
        return self.data.body(self.end_platform_id).cvel[3:6].copy()
    
    def get_ball_pos(self) -> np.ndarray:
        """
        Returns ball position in world frame from sensor.
        
        Returns:
            np.ndarray: Position [x, y, z] in world frame (3,).
        """
        if self.ball_pos_sensor_id != -1:
            adr = self.model.sensor_adr[self.ball_pos_sensor_id]
            dim = self.model.sensor_dim[self.ball_pos_sensor_id]
            return self.data.sensordata[adr:adr+dim].copy()
        return self.data.body(self.ball_id).xpos.copy()
    
    def get_ball_vel(self) -> np.ndarray:
        """
        Returns ball linear velocity in world frame from sensor.
        
        Returns:
            np.ndarray: Velocity [vx, vy, vz] in world frame (3,).
        """
        if self.ball_lin_vel_sensor_id != -1:
            adr = self.model.sensor_adr[self.ball_lin_vel_sensor_id]
            dim = self.model.sensor_dim[self.ball_lin_vel_sensor_id]
            return self.data.sensordata[adr:adr+dim].copy()
        # Fallback: get from body velocity
        return self.data.body(self.ball_id).cvel[3:6].copy()
    
    def set_ball_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """
        Set ball initial position and velocity.
        Must be called before simulation starts.
        
        Args:
            pos: Position [x, y, z] in world frame.
            vel: Velocity [vx, vy, vz] in world frame.
        """
        # Set position via qpos
        ball_qpos_adr = self.model.jnt_qposadr[self.model.body_jntadr[self.ball_id]]
        self.data.qpos[ball_qpos_adr:ball_qpos_adr+3] = pos
        self.data.qpos[ball_qpos_adr+3:ball_qpos_adr+7] = [1, 0, 0, 0]  # Identity quaternion
        
        # Set velocity via qvel
        ball_qvel_adr = self.model.jnt_dofadr[self.model.body_jntadr[self.ball_id]]
        self.data.qvel[ball_qvel_adr:ball_qvel_adr+3] = vel
        self.data.qvel[ball_qvel_adr+3:ball_qvel_adr+6] = [0, 0, 0]  # Zero angular velocity
        
        # Forward kinematics to update sensor data
        mujoco.mj_forward(self.model, self.data)
    
    def reset_ball_on_ground(self) -> None:
        """
        Reset ball to rest on ground (stop simulation for this ball).
        Sets velocity to zero and keeps ball at current ground position.
        """
        ball_pos = self.get_ball_pos()
        ball_vel = np.zeros(3)
        
        # Set position slightly above ground
        if ball_pos[2] < 0.05:
            ball_pos[2] = 0.03  # Rest on ground
        
        self.set_ball_state(ball_pos, ball_vel)
    
    def print_ball_state(self) -> None:
        """Print ball state."""
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        print(f"Ball Position (world): {self.get_ball_pos()}")
        print(f"Ball Velocity (world): {self.get_ball_vel()}")
    
    def print_ee_state(self) -> None:
        """Print end-effector state with sensor offset."""
        np.set_printoptions(formatter={'float': '{: .4f}'.format})
        print(f"EE Sensor Position (base-relative, offset={self.sensor_offset_z*100:.1f}cm): {self.get_ee_sensor_pos()}")
        print(f"EE Velocity: {self.get_ee_sensor_lin_vel()}")
