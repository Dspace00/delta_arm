import mujoco
import mujoco.viewer
import numpy as np
import time 
import os

from DeltaKinematics import DeltaKinematics

class SimpleController:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.running = True 
        self._setup_model_indices()
        self.L = 0.1
        self.l = 0.2
        self.R = 0.074577
        self.r = 0.02495

    def _setup_model_indices(self):
        """设置模型组件索引"""
        # 执行器索引
        self.motor_actuator_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor1_vel'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor2_vel'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor3_vel')
        ]
        
        # 关节索引
        self.motor_joint_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'motor_arm_joint1'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'motor_arm_joint2'), 
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'motor_arm_joint3')
        ]
        
        # 关节位置地址
        self.motor_qpos_adr = [self.model.joint(i).qposadr[0] for i in self.motor_joint_indices]
        
         # 关节速度地址
        self.motor_qvel_adr = [self.model.joint(i).dofadr[0] for i in self.motor_joint_indices]

        # 末端平台body ID
        self.end_platform_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'end_platform')

    def get_joint_angles(self) -> np.ndarray:
        """获取当前关节角度"""
        return np.array([self.data.qpos[adr] for adr in self.motor_qpos_adr])

    def get_joint_velocities(self) -> np.ndarray:
        """获取关节角速度"""
        return np.array([self.data.qvel[adr] for adr in self.motor_qvel_adr])

    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置"""
        return self.data.xpos[self.end_platform_id].copy()
    
    def get_end_effector_velocity(self) -> np.ndarray:
        """获取末端执行器速度"""
        return self.data.cvel[self.end_platform_id][3:6].copy()

    def get_status(self) -> dict:
        """获取当前状态数据"""
        joint_angles_rad = self.get_joint_angles()
        joint_angles_deg = np.degrees(joint_angles_rad)
        joint_velocities = self.get_joint_velocities()
        ee_position = self.get_end_effector_position()
        ee_velocity = self.get_end_effector_velocity()
        print(f"Joint Angles (rad): {[f'{v:.5f}' for v in joint_angles_rad]}")
        print(f"Joint Angles (deg): {[f'{v:.5f}' for v in joint_angles_deg]}")
        print(f"Joint Velocities (rad/s): {[f'{v:.5f}' for v in joint_velocities]}")
        print(f"End Effector Position (m): {[f'{v:.5f}' for v in ee_position]}")
        print(f"End Effector Velocity (m/s): {[f'{v:.5f}' for v in ee_velocity]}")

    def velocity_control(self, target_velocities):
        self.data.ctrl[self.motor_actuator_indices[0]] = target_velocities[0]
        self.data.ctrl[self.motor_actuator_indices[1]] = target_velocities[1]   
        self.data.ctrl[self.motor_actuator_indices[2]] = target_velocities[2]

class PositionMove:
    def __init__(self, start_pos, target_pos, total_steps, sim_timestep):
        """
        基于步数的位置移动控制器
        参数:
        start_pos: 起始位置 [x, y, z]
        target_pos: 目标位置 [x, y, z]  
        total_steps: 总步数
        """
        self.start_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)
        self.total_steps = total_steps
        self.current_step = 0
        self.sim_timestep = sim_timestep
        
        # 计算每步的位移
        self.displacement = self.target_pos - self.start_pos
        self.step_displacement = self.displacement / self.total_steps if self.total_steps > 0 else np.zeros(3)
        
        # 计算恒定速度（每步的位移除以仿真步长）
        self.velocity = self.step_displacement / self.sim_timestep if self.total_steps > 0 else np.zeros(3)
        
        self.current_pos = self.start_pos.copy()
        self.is_completed = False

    def get_state_at_step(self, step):
        """
        获取指定步数的状态
        
        参数:
        step: 当前仿真步数
        
        返回:
        position: 当前位置
        velocity: 当前速度
        is_completed: 是否完成运动
        """
        if self.is_completed:
            return self.target_pos.copy(), np.zeros(3), True
            
        if step >= self.total_steps:
            self.is_completed = True
            self.current_pos = self.target_pos.copy()
            return self.target_pos.copy(), np.zeros(3), True
        
        # 计算当前位置
        progress = step / self.total_steps
        self.current_pos = self.start_pos + progress * self.displacement
        
        return self.current_pos.copy(), self.velocity.copy(), False

    def reset(self):
        """重置运动状态"""
        self.current_step = 0
        self.current_pos = self.start_pos.copy()
        self.is_completed = False

    def get_progress(self):
        """获取运动进度 (0到1之间)"""
        if self.is_completed:
            return 1.0
        return min(self.current_step / self.total_steps, 1.0)

    def get_remaining_steps(self):
        """获取剩余步数"""
        if self.is_completed:
            return 0
        return max(self.total_steps - self.current_step, 0)

def main():
    controller = SimpleController("urdf/Delta.xml")
    controller.model.opt.integrator = 3 # 使用RK4积分器
    sim_timestep_count = 0
    sim_timestep = controller.model.opt.timestep
    robot = DeltaKinematics()
    # 控制参数
    control_frequency = 100  # 控制频率 (Hz)
    num_steps_per_control = int(1 / (sim_timestep * control_frequency))
    
    # 运动参数 - 基于步数而不是时间
    target_pos = [0.0, 0.0, -0.095]
    motor_angle = robot.ik(target_pos)


    # motor_angle_start = robot.inverse_kinematics(start_pos)
    total_movement_steps = 500  # 总运动步数


    def get_status_at_step(step):
        """获取指定步数的状态"""
        pos = controller.get_end_effector_position()
        vel = controller.get_end_effector_velocity()
        des_pos, des_vel, is_completed = position_move.get_state_at_step(step)
        print(f"Step {step}: Desired Position: {des_pos}, Desired Velocity: {des_vel}")
        return pos, vel, des_pos, des_vel, is_completed

    def motor_angular_velocity_from_end_platform_and_desired_trajectory(pos, vel, des_pos, des_vel):
        """根据末端位置和期望轨迹计算电机角速度"""
        L_upper = controller.L
        L_lower = controller.l
        R = controller.R
        r = controller.r

        # PD控制器计算速度修正
        err_pos = des_pos - pos
        err_vel = des_vel - vel
        kp = 2
        kd = 1  
        vel_correct = kp * err_pos + kd * err_vel
        vel_control = des_vel + vel_correct

        def phi_matrix(i):
            """计算旋转矩阵 phi_i (公式7)"""
            angle = i * np.pi / 3  # (i)π/3，这里i从0开始
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            phi_i = np.array([
                [cos_angle, sin_angle, 0],
                [-sin_angle, cos_angle, 0],
                [0, 0, 1]
            ])
            return phi_i
        
        def alpha_vector(q_i):
            """计算alpha向量 (公式9)"""
            return np.array([np.cos(q_i), 0, np.sin(q_i)])
        
        def d_alpha_dq(q_i):
            """计算alpha向量对q_i的导数"""
            return np.array([-np.sin(q_i), 0, np.cos(q_i)])
        
        def e_vector(q_i, i):
            """计算e_i向量 (公式9)"""
            alpha_i = alpha_vector(q_i)
            base_vector = np.array([R - r, 0, 0])
            e_i = base_vector + L_upper * alpha_i
            return e_i
        
        def beta_vector(pos, q_i, i):
            """计算beta向量 (公式8)"""
            phi_i = phi_matrix(i)
            p_Ei_e = phi_i @ pos  # 公式6
            e_i = e_vector(q_i, i)
            beta_i = (p_Ei_e - e_i) / L_lower
            return beta_i
                
        motor_angle = controller.get_joint_angles().tolist()
        
        # 初始化M和V矩阵
        M_rows = []
        V_diag = []
        
        for i in range(3):
            # 计算beta_i
            beta_i = beta_vector(pos, motor_angle[i], i)
            
            # 计算phi_i
            phi_i = phi_matrix(i)
            
            # 计算M矩阵的行 (公式12)
            M_row = beta_i.T @ phi_i
            M_rows.append(M_row)
            
            # 计算d_alpha/dq
            d_alpha_dq_val = d_alpha_dq(motor_angle[i])
            
            # 计算V矩阵的对角元素 (公式12)
            V_diag_element = beta_i.T @ d_alpha_dq_val
            V_diag.append(V_diag_element)
        
        # 构建M和V矩阵
        M = np.array(M_rows)
        V = np.diag(V_diag)
        
        # 计算电机角速度 (从公式11推导)
        # M * p_E_dot_e = L_upper * V * q_dot
        # => q_dot = (1/L_upper) * V^(-1) * M * p_E_dot_e
        
        try:
            V_inv = np.linalg.inv(V)
            q_dot = (1 / L_upper) * V_inv @ M @ vel_control
        except np.linalg.LinAlgError:
            # 如果V矩阵奇异，使用伪逆
            V_inv = np.linalg.pinv(V)
            q_dot = (1 / L_upper) * V_inv @ M @ vel_control
        
        print(f"q_dot: {q_dot}")
        return q_dot

    def print_states():
        """打印当前状态"""
        os.system('cls' if os.name == 'nt' else 'clear')
        # print(f"motor_angle_from_ik: {motor_angle}")
        print(f"target_pos: {target_pos}")
        print(f"start_pos: {start_pos}")
        controller.get_status()
        print(f"sim_timestep: {sim_timestep} seconds")
        print(f"actual sim time: {sim_timestep_count * sim_timestep:.5f} seconds")
        print(f"当前仿真步数: {sim_timestep_count}")
        print(f"运动进度: {position_move.get_progress():.3f}")
        print(f"剩余步数: {position_move.get_remaining_steps()}")

    # 主仿真循环
    with mujoco.viewer.launch_passive(
        controller.model, controller.data
    ) as viewer:
        print("MuJoCo查看器已启动")
        start_time = time.time()
        start_pos = controller.get_end_effector_position()     
        position_move = PositionMove(start_pos, target_pos, total_movement_steps, sim_timestep)  
        while viewer.is_running and controller.running:
            # os.system('cls' if os.name == 'nt' else 'clear')
            # start_pos = controller.get_end_effector_position()
            # motor_angle_ik = np.array(robot.ik(start_pos))
            # rad_ik = np.deg2rad(motor_angle_ik)
 
            # motor_angle_form_mea = np.rad2deg(controller.get_joint_angles())
            
            # print(f"startpos_shape: {start_pos.shape}")
            # print(f"start_pos: {[f'{v:.5f}' for v in start_pos]}")
            # print(f"angle_ik in deg: {[f'{v:.5f}' for v in motor_angle_ik]}")
            # print(f"angle_ik in rad: {[f'{v:.5f}' for v in rad_ik]}")
            # print(f"measurement in deg: {[f'{v:.5f}' for v in motor_angle_form_mea]}")
            # print(f"measurement in rad: {[f'{v:.5f}' for v in controller.get_joint_angles()]}")
            # print(f"degree_ik: {[f'{v:.5f}' for v in degree_ik]}")
        
            
            # print(f"error percentage: {angle_eeror}")
            # print(angle_eeror.shape)
            # 获取当前状态

            
            # 执行仿真步
            position_move.get_state_at_step(sim_timestep_count)
            print(f"current_des_pos: {position_move.current_pos}")
            # # 控制更新
            if sim_timestep_count % num_steps_per_control == 0:
                pos, vel, des_pos, des_vel, is_completed = get_status_at_step(sim_timestep_count)
            
                # # 计算电机速度
                motor_velocities = motor_angular_velocity_from_end_platform_and_desired_trajectory(pos, vel, des_pos, des_vel)
                last_motor_velocity = motor_velocities
                if is_completed and np.linalg.norm(pos - des_pos) < 1e-3:
                    motor_velocities = np.zeros(3)
                    last_motor_velocity = motor_velocities
                    controller.velocity_control(last_motor_velocity)
                    if is_completed:
                        print("到达目标位置，停止运动")
                else:   
                    controller.velocity_control(motor_velocities)
            else:        
                controller.velocity_control(last_motor_velocity)
            mujoco.mj_step(controller.model, controller.data)
            viewer.sync()
            sim_timestep_count += 1
            print_states()

if __name__ == "__main__":
    main()