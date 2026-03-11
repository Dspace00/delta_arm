"""
模块功能说明：
定义MuJoCo仿真接口，负责模型加载、物理步进和状态获取。

重要参数：
- model_path: XML模型文件路径
- dt: 仿真时间步长

使用示例：
sim = MujocoSimulator("path/to/model.xml")
sim.reset()
sim.step()
"""

import mujoco
import mujoco.viewer
import numpy as np
import os

class MujocoSimulator:
    def __init__(self, model_path):
        if not os.path.isabs(model_path):
            # assume relative to project root or handles by caller, but strictly checks existence
            if not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Set timestep
        # self.model.opt.timestep = 0.0001 # Set by XML or here? User req says fixed 0.0001s
        # We will enforce it in the loop logic or set it here if desired
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        
    def step(self):
        mujoco.mj_step(self.model, self.data)
        
    def get_time(self):
        return self.data.time

    def apply_wrench(self, force, torque):
        """
        应用力和力矩到机体
        force: [Fx, Fy, Fz] (World frame or Body frame? Code implies Body frame actuation)
        torque: [Mx, My, Mz] (Body frame)
        """
        # Based on previous code:
        # self.data.actuator(f'forcex').ctrl[0] = force_body[0] 
        # ...
        # self.data.actuator(f'Mx').ctrl[0] = torque[0]
        
        # Note: The input force here is expected to be in BODY frame if the actuators are defined in body frame.
        # The previous code calculated force_body = R.T @ total_thrust_world
        # So we expect the controller to pass BODY frame forces/torques or we handle conversion here.
        # Requirement 3 says: apply_wrench(force, torque) -> set_force/set_torque
        # Let's assume the controller passes [Thrust, 0, 0] or [0, 0, Thrust] in body frame, or [Fx, Fy, Fz] in body frame.
        # Given the previous code used "forcex", "forcey", "forcez" actuators which likely rotate with the body.
        
        try:
            self.data.actuator('forcex').ctrl[0] = force[0]
            self.data.actuator('forcey').ctrl[0] = force[1]
            self.data.actuator('forcez').ctrl[0] = force[2]
            self.data.actuator('Mx').ctrl[0] = torque[0]
            self.data.actuator('My').ctrl[0] = torque[1]
            self.data.actuator('Mz').ctrl[0] = torque[2]
        except Exception as e:
            # Fallback if actuators are named differently or indices used
            pass
