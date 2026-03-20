# `gym-pybullet-drones` 整体架构、精细建模与控制器实现深度解析

## 1. 简介与背景
`gym-pybullet-drones` 是一个基于 PyBullet 的无人机强化学习和控制仿真环境。与基础的刚体仿真器不同，它不仅提供了多种传统控制算法，还通过**手动施加外力**的方式实现了精细的空气动力学建模（地面效应、下洗气流、阻力等）。同时，它将底层物理引擎完美封装为标准的 Gymnasium 接口，支持多种动作/状态空间组合。

本文档不仅从宏观架构上分析该项目，还深入到**具体代码实现层面**，并对比当前 `Project_uav`（基于 MuJoCo）的架构，给出详细的移植与重构代码示例。

---

## 2. 环境层与精细物理建模分析 (`envs/BaseAviary.py`)

标准物理引擎（如 PyBullet、MuJoCo）原生只处理刚体碰撞和基础重力。为了让无人机飞得“像真的一样”，`BaseAviary` 在每次物理步进（`stepSimulation`）之前，通过 `p.applyExternalForce` 和 `p.applyExternalTorque` 手动注入空气动力学干涉。

### 2.1 基础推力与扭矩模型 (`_physics`)
无人机的四个电机分别产生向上的推力和反扭矩，这是基于转速的平方（$\text{RPM}^2$）计算的。

**具体代码实现：**
```python
def _physics(self, rpm, nth_drone):
    # 根据 RPM 计算推力和扭矩 (F = kf * RPM^2, tau = km * RPM^2)
    forces = np.array(rpm**2) * self.KF
    torques = np.array(rpm**2) * self.KM
    
    # 旋翼反扭矩抵消计算（通常对角线电机转向相同）
    z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
    
    # 将推力施加到 0~3 号 link（即四个电机的位置）上
    for i in range(4):
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             i,
                             forceObj=[0, 0, forces[i]],
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT)
    # 将反扭矩施加到无人机质心（Link 4 或 Base）
    p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                          4,
                          torqueObj=[0, 0, z_torque],
                          flags=p.LINK_FRAME,
                          physicsClientId=self.CLIENT)
```

### 2.2 地面效应 (Ground Effect)
当无人机非常靠近地面时，向下排出的气流会被地面反弹，导致实际升力变大。

**具体代码实现：**
```python
def _groundEffect(self, rpm, nth_drone):
    # 1. 获取每个螺旋桨距离地面的高度
    link_states = p.getLinkStates(...)
    prop_heights = np.array([link_states[0][0][2], link_states[1][0][2], ...])
    
    # 2. 限制最低高度以防止除以零或无穷大
    prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
    
    # 3. 计算地面效应带来的额外升力 (与 (R / 4h)^2 成正比)
    gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS / (4 * prop_heights))**2
    
    # 4. 只有在姿态不倾斜过大（<90度）时才施加该力
    if np.abs(self.rpy[nth_drone,0]) < np.pi/2 and np.abs(self.rpy[nth_drone,1]) < np.pi/2:
        for i in range(4):
            p.applyExternalForce(..., forceObj=[0, 0, gnd_effects[i]], flags=p.LINK_FRAME)
```

### 2.3 空气阻力 (Drag)
基于无人机的线速度和螺旋桨转速计算的阻力。

**具体代码实现：**
```python
def _drag(self, rpm, nth_drone):
    # 将世界坐标系下的速度转换到机体坐标系，或者通过旋转矩阵作用于阻力系数
    base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
    
    # 阻力大小与转速之和及当前线速度成正比
    drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2 * np.pi * rpm / 60))
    drag = np.dot(base_rot.T, drag_factors * np.array(self.vel[nth_drone, :]))
    
    # 施加阻力于无人机质心
    p.applyExternalForce(..., forceObj=drag, flags=p.LINK_FRAME)
```

---

## 3. 强化学习架构封装 (`envs/BaseRLAviary.py`)

`BaseRLAviary` 的核心是将上述复杂的连续控制引擎降维/规范化为 RL 智能体可以理解的 `Box` 空间。

### 3.1 状态空间 (Observation Space) 与动作延迟缓冲
强化学习非常忌讳“不可见的状态”（如电机的响应延迟）。为了让马尔可夫决策过程 (MDP) 成立，框架会将**过去的动作历史**拼接到当前状态中。

**具体代码实现：**
```python
def _computeObs(self):
    # 1. 提取当前运动学状态（长度为12）: [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz]
    obs_12 = np.zeros((self.NUM_DRONES, 12))
    for i in range(self.NUM_DRONES):
        obs = self._getDroneStateVector(i)
        obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]])
    
    ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
    
    # 2. 拼接 Action Buffer (默认长度为半秒内的动作)，让神经网络“知道”自己之前下达的指令，从而推断延迟
    for i in range(self.ACTION_BUFFER_SIZE):
        ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
        
    return ret
```

### 3.2 动作空间映射 (`_preprocessAction`)
直接输出 RPM 对神经网络来说太难学习。框架支持让 RL 网络输出“期望位置”，然后在环境内部运行传统的 PID 控制器去跟踪。

**具体代码实现：**
```python
def _preprocessAction(self, action):
    # action 形状为 (NUM_DRONES, action_dim)
    self.action_buffer.append(action)
    rpm = np.zeros((self.NUM_DRONES, 4))
    
    for k in range(action.shape[0]):
        target = action[k, :]
        
        # 模式1: RL 直接输出目标 RPM
        if self.ACT_TYPE == ActionType.RPM:
            rpm[k,:] = np.array(self.HOVER_RPM * (1 + 0.05 * target))
            
        # 模式2: RL 输出目标航点，环境内部用 PID 控制器进行底层解算
        elif self.ACT_TYPE == ActionType.PID:
            state = self._getDroneStateVector(k)
            # 计算中间目标航点，防止跨度过大导致 PID 崩溃
            next_pos = self._calculateNextStep(current_position=state[0:3], destination=target)
            # 调用内置的 DSLPIDControl
            rpm_k, _, _ = self.ctrl[k].computeControl(..., cur_pos=state[0:3], target_pos=next_pos)
            rpm[k,:] = rpm_k
            
    return rpm
```

---

## 4. 控制器源码深度剖析 (`control/`)

### 4.1 `DSLPIDControl` (位置与姿态级联 PID)
这是最典型的多旋翼控制算法。代码分为两部分：位置控制解算出期望推力与目标姿态，姿态控制解算出三轴扭矩。

**具体代码实现 (位置控制提取期望推力与姿态)：**
```python
def _dslPIDPositionControl(self, ...):
    # 1. 计算 PID 输出的目标加速度（Target Thrust）
    pos_e = target_pos - cur_pos
    vel_e = target_vel - cur_vel
    self.integral_pos_e += pos_e * control_timestep
    
    target_thrust = (np.multiply(self.P_COEFF_FOR, pos_e) +
                     np.multiply(self.I_COEFF_FOR, self.integral_pos_e) +
                     np.multiply(self.D_COEFF_FOR, vel_e) + 
                     np.array([0, 0, self.GRAVITY])) # 补偿重力
                     
    # 2. 标量推力：提取目标推力在当前机体 Z 轴方向的投影
    scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
    thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
    
    # 3. 计算期望的姿态 (将机体 Z 轴对齐到目标推力方向)
    target_z_ax = target_thrust / np.linalg.norm(target_thrust)
    target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0]) # 目标偏航
    target_y_ax = np.cross(target_z_ax, target_x_c)
    target_y_ax /= np.linalg.norm(target_y_ax)
    target_x_ax = np.cross(target_y_ax, target_z_ax)
    
    target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
    target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ')
    
    return thrust, target_euler, pos_e
```

### 4.2 `MRAC` (模型参考自适应控制)
MRAC 的核心在于应对外部扰动和质量变化。代码内部同时维护了一个**理想数学模型 (`Am, Bm`)** 和一套自适应更新律 (`Kx, Kr`)。

**具体代码实现 (李雅普诺夫方程与增益更新)：**
```python
def computeControl(self, ...):
    # 1. 计算期望状态和控制输入
    rt = -self.Kr_ref_gain @ r # r为包含目标位置、姿态的12维向量
    X_actual = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel)).reshape(12, 1)
    
    # 当前控制律 u = Kx * X + Kr * r
    u = self.Kx.T @ X_actual + self.Kr.T @ rt
    
    # 2. 计算实际状态与参考模型(Xm)的误差
    e = X_actual - self.Xm 
    
    # 3. 基于李雅普诺夫稳定性的自适应律：动态更新增益 Kx 和 Kr
    # Kx_dot = -Gamma_x * X * e^T * P * Bm
    Kx_dot = -self.Gamma_x @ X_actual @ e.T @ self.P @ self.Bm
    Kr_dot = -self.Gamma_r @ rt @ e.T @ self.P @ self.Bm
    self.Kx += Kx_dot * control_timestep
    self.Kr += Kr_dot * control_timestep
    
    # 4. 推进参考模型 (Reference Model)
    Xm_dot = self.Am @ self.Xm + self.Bm @ rt
    self.Xm += Xm_dot * control_timestep
    
    # 5. 解算 u (Thrust, Tx, Ty, Tz) 为 RPM ...
```

---

## 5. `Project_uav` (MuJoCo) 架构对比与移植重构指南

目前 `Project_uav` 依赖 `mujoco.mj_step` 进行仿真，控制逻辑在 `CascadeController` 中由 PyTorch 张量实现。

### 5.1 补齐精细物理建模 (地面效应等)
PyBullet 通过 `applyExternalForce` 施加力，在 MuJoCo 中等价的方法是直接向 `data.xfrc_applied` 注入外力，或使用 `mj_applyFT`。

**在 `Project_uav/simulator.py` 中补充建模的代码示例：**
```python
def _step_simulation(self, step, trajectory):
    sim_time = step * self.timestep
    
    # 1. Update Target & Controller ...
    self.controller.update(sim_time)
    
    # ================= 新增：空气动力学注入 =================
    # 获取无人机质心高度 (假设 uav_body_id 为无人机的机体 ID)
    uav_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'uav_body')
    z_height = self.data.xpos[uav_body_id][2]
    
    # 计算地面效应附加推力 (需根据实际 RPM 和系数计算)
    if z_height < 0.5: # 在 0.5m 内产生显著地面效应
        # 假设从控制器获取了当前的推力系数和 RPM 估算
        # f_gnd = kf * rpm^2 * coeff * (R / 4z)^2
        f_gnd = self.controller.estimate_ground_effect(z_height)
        
        # 向机体施加垂直向上的外力
        # mj_applyFT (model, data, force, torque, point, body_id, qfrc_target)
        mujoco.mj_applyFT(self.model, self.data, 
                          np.array([0.0, 0.0, f_gnd]), # 力
                          np.array([0.0, 0.0, 0.0]),   # 扭矩
                          self.data.xpos[uav_body_id], # 施力点(质心)
                          uav_body_id, 
                          self.data.qfrc_applied)
    # ========================================================
    
    # 2. Step Physics
    mujoco.mj_step(self.model, self.data)
```

### 5.2 强化学习 (RL) 接口重构
当前 `Project_uav` 的 `run()` 方法是一个一次性执行到底的循环，无法被 RL 算法（如 PPO/SAC）调用。我们需要创建一个 `gym.Env` 包装器。

**`Project_uav/envs/uav_rl_env.py` 设计蓝图：**
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UAVEnv(gym.Env):
    def __init__(self, simulator):
        super().__init__()
        self.sim = simulator
        
        # 定义动作空间: 输出期望位置增量 (X, Y, Z, Yaw)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 定义状态空间: 12维状态 + 历史动作
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        
    def reset(self, seed=None):
        mujoco.mj_resetData(self.sim.model, self.sim.data)
        self.sim.controller.reset()
        return self._get_obs(), {}
        
    def step(self, action):
        # 1. 预处理动作 (如: 当前位置 + 缩放后的动作 = 目标位置)
        current_pos = self.sim.data.qpos[:3]
        target_pos = current_pos + action[:3] * 0.1
        self.sim.controller.set_target_position(target_pos)
        
        # 2. 执行一次控制与物理步进 (控制频率可低于物理频率，此处简化)
        self.sim.controller.update(self.sim.data.time)
        mujoco.mj_step(self.sim.model, self.sim.data)
        
        # 3. 获取观测与计算奖励
        obs = self._get_obs()
        reward = self._compute_reward(current_pos, target_pos)
        terminated = self._check_crash()
        
        return obs, reward, terminated, False, {}
        
    def _get_obs(self):
        # 提取 MuJoCo 状态并拼接返回
        # ...
```

### 5.3 控制器移植：动态惯性张量读取
在 `gym-pybullet-drones` 的 `MRAC` 中，控制律极其依赖于 $I_{xx}, I_{yy}, I_{zz}$。在 `Project_uav` 现有的 `cascade_controller.py` 中，我们可以直接通过 MuJoCo 的 API 获取精确的运行时物理属性。

**在 `cascade_controller.py` 的初始化中补充：**
```python
# 获取 UAV_body 对应的 ID
body_id = mujoco.mj_name2id(self.uav.model, mujoco.mjtObj.mjOBJ_BODY, 'uav_body')

# 读取质量和对角线惯性张量 (Ixx, Iyy, Izz)
self.mass = self.uav.model.body_mass[body_id]
self.inertia = self.uav.model.body_inertia[body_id] 
# self.inertia 返回形如 [Ixx, Iyy, Izz] 的数组

# 这样在移植 MRAC 或进行姿态解耦控制时，就可以直接使用 self.inertia 进行前馈补偿
```