"""
MuJoCo仿真环境封装
"""
import mujoco
import mujoco.viewer
import numpy as np
from typing import Optional, Tuple
import time


class MuJoCoEnvironment:
    """MuJoCo仿真环境"""
    
    def __init__(self, model_path: str, dt: Optional[float] = None):
        """
        初始化MuJoCo环境
        
        Args:
            model_path: MuJoCo模型文件路径
            dt: 仿真时间步长，如果为None则使用模型默认值
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 时间参数
        self.sim_dt = dt if dt is not None else self.model.opt.timestep
        self.sim_time = 0.0
        self.step_count = 0
        
        # 渲染设置
        self.viewer = None
        self.render_enabled = False
        
    def reset(self):
        """重置仿真"""
        mujoco.mj_resetData(self.model, self.data)
        self.sim_time = 0.0
        self.step_count = 0
        print("Simulation reset")
        
    def step(self, control_inputs: Optional[dict] = None):
        """
        执行一个仿真步
        
        Args:
            control_inputs: 控制输入字典，键为执行器名称，值为控制值
        """
        # 应用控制输入
        if control_inputs:
            for actuator_name, value in control_inputs.items():
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id != -1:
                    self.data.ctrl[actuator_id] = value
        
        # 执行仿真步
        mujoco.mj_step(self.model, self.data)
        
        # 更新状态
        self.sim_time += self.sim_dt
        self.step_count += 1
        
    def get_body_state(self, body_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取指定刚体的状态
        
        Args:
            body_name: 刚体名称
            
        Returns:
            position: 位置 [x, y, z]
            quaternion: 四元数 [w, x, y, z]
            velocity: 线速度 [vx, vy, vz]
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model")
        
        # 位置和四元数
        position = self.data.body(body_id).xpos.copy()
        quaternion = self.data.body(body_id).xquat.copy()  # [w, x, y, z]
        
        # 线速度（世界坐标系）
        velocity = self.data.body(body_id).cvel[3:6].copy()
        
        return position, quaternion, velocity
    
    def launch_viewer(self):
        """启动MuJoCo查看器"""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.render_enabled = True
        print("MuJoCo viewer launched")
        
    def close_viewer(self):
        """关闭查看器"""
        if self.viewer is not None:
            self.viewer.close()
            self.render_enabled = False
            
    def sync_viewer(self):
        """同步查看器（如果启用）"""
        if self.render_enabled and self.viewer is not None:
            self.viewer.sync()
            
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close_viewer()