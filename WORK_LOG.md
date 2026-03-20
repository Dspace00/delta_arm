# UAV Project Development Log

## 2026-03-19

### 1. 控制器 PyTorch 张量化重构 (Controller PyTorch Refactoring)
- 将核心控制器的底层运算从 NumPy 迁移至 PyTorch 张量 (Tensor)，涉及文件包括：
  - `uav_project/controllers/cascade_controller.py`
  - `uav_project/controllers/delta_controller.py`
  - `uav_project/controllers/pid.py`
- 统一矩阵运算形状（如明确 `(3, 1)` 列向量），并引入 `torch.no_grad()` 优化前向控制计算的性能。
- 修复了 NumPy 数组与 PyTorch 张量混合运算导致的 `TypeError` 问题，保证了端到端张量计算。

### 2. 数学函数与运动学库统一 (Math Utilities Standardization)
- 对项目中的基础数学运算和运动学库进行了 PyTorch 原生化重构，涉及文件：
  - `uav_project/utils/SimpleMath.py`
  - `uav_project/utils/DeltaKinematics.py`
  - `uav_project/utils/trajectory.py`
- 将原有的 `math.sin`、`math.cos`、`math.pi` 等替换为原生的 `torch.sin`、`torch.cos`、`torch.pi`，确保整个控制和轨迹生成链路的数据类型及设备一致性，进一步提升计算效率。
- 在自定义技能 `matrix-control` 中增加了处理数学函数和常数的约束，确保后续代码的张量计算标准。

### 3. SSH 远程开发无头模式适配 (SSH Headless Mode Adaptation)
- 针对 SSH 远程开发环境下无法调用显示器 (X11 DISPLAY 报错) 的问题，在 `uav_project/main.py` 中为 MuJoCo 仿真开启了 `headless=True`。
- 修改了 `uav_project/utils/logger.py`，将 Matplotlib 的 backend 设置为 `Agg`，使得仿真绘图结果可以直接无界面保存为本地图片 (`simulation_results.png`)。

### 4. 环境配置与文档规范化 (Environment & Documentation)
- 统一了开发环境，明确所有后续开发任务使用 `mujoco-sim` conda 虚拟环境。
- 更新了项目根目录的 `/environment.yml`，明确加入了核心依赖项 `torch>=2.0.0` 和 `mujoco`。
- 全面更新了 `uav_project/readme.md`，加入了详细的新手配置指南、依赖安装步骤以及运行方式说明，形成闭环的 Onboarding 文档。

## 2026-03-20

### 1. 推力计算与物理参数动态绑定
- **动态质量读取**：修复了 Delta 模型下无人机推力不足无法起飞的问题。将 `cascade_controller.py` 中硬编码的 `MASS` 修改为通过 `sum(self.uav.model.body_mass)` 动态读取 MuJoCo 模型所有刚体质量的总和，确保无论是否挂载机械臂，前馈推力都能正确计算。
- **电机参数解耦计算**：在 `config.py` 中，不再硬编码 $C_T$ 和 $C_D$。改为由 `MAX_THRUST_PER_MOTOR`、`MAX_TORQUE_PER_MOTOR` 和 `MAX_MOTOR_SPEED_KRPM` 动态计算出对应的 $C_T$ 和 $C_D$。
- **底层物理引擎参数同步**：在 `uav_model.py` 的初始化逻辑中，加入了动态覆盖 MuJoCo XML 中 `motor` 驱动器 `gear` 比例的逻辑。使得底层仿真物理引擎严格与 `config.py` 中的数学模型对齐。

### 2. Delta 机械臂运动学公式与坐标系说明文档
- 修复了 `DeltaKinematics.py` 中的轨迹跟踪问题：恢复了故意加入的符号反转和 90 度坐标系旋转矩阵，以适配 MuJoCo 中的坐标系定义。
- 编写了详尽的 `DELTA_KINEMATICS_MATH.md`，使用 LaTeX 公式完整推导了 Delta 机械臂的逆运动学（IK）和正运动学（FK），并特别记录了代码实现中的“折中策略”（如为了适配底层 XML 定义而故意导致的 IK/FK 不闭环问题）。

### 3. 数据可视化 (Logger) 与终端打印深度优化
- **自适应终端打印**：修改 `uav_model.py` 的打印逻辑，如果加载的是无机械臂模型 (`UAV.xml`)，则自动隐藏关于末端平台的冗余全 0 数据。
- **PID 串级分层可视化**：重构 `logger.py` 的 Matplotlib 绘图布局。现在生成的 `simulation_results.png` 严格按照串级 PID 的层级排列，依次展示：位置(XYZ)、速度(Vx/Vy/Vz)、姿态欧拉角(Roll/Pitch/Yaw)、角速度(P/Q/R)、控制输出(推力F/力矩M) 以及 四个电机的单独推力分配。极大提升了调试效率。

### 4. 隔离排查：四转子混控模式 vs 质心直接控制模式
为了排查无人机在引入姿态控制后出现的抖动与不稳定现象，进行了控制模式的隔离测试：
- **四转子电机混控模式 (Mixer Control)**：控制流为 `PID -> F/M -> Mixer逆解 -> 4个电机 kRPM^2 -> MuJoCo rotor 驱动器`。这是实际物理机的控制方式，但目前表现不稳定。
- **质心直接控制模式 (Direct Force Control)**：修改 XML 开启 `forcex`~`Mz` 驱动器，修改代码直接将 PID 算出的 F/M 发送给质心，跳过 Mixer。
- **测试结论**：在“质心直接控制模式”下，无人机能够稳定追踪轨迹（尽管响应略慢）。这**证明了外环 PID 控制器（位置、速度、姿态、角速度）的大致逻辑与参数是正确的**。
- **下一步方向**：确认导致不稳定的核心问题出在内环的 **Mixer 混控解算**、**电机参数映射**或 **XML 驱动器定义（如扭矩正负号方向与 Mixer 矩阵不匹配）** 上。目前已将代码回退为“四转子电机混控模式”以便集中修复 Mixer 问题。

### 5. 修复姿态控制逻辑与跨模型 PID 参数自适应
- **修复机体系四元数误差计算**：在 `controllers/pid.py` 的 `AttitudePID3D` 中，将世界系下的旋转误差计算 `quat_error = tar_q * cur_conj` 修正为了机体系下的标准 SO(3) 误差计算公式 `quat_error = cur_conj * tar_q`，彻底解决了无人机发生偏航后姿态补偿方向错误导致“倒扣”炸机的致命 Bug。
- **解决挂载 Delta 机械臂后的飘逸问题**：
  - 修复了 `cascade_controller.py` 中动态质量的计算逻辑，通过 `sum(m for m in self.uav.model.body_mass if m > 0)` 过滤了 `worldbody` 和无质量刚体，确保系统能够获取最准确的总挂载质量（纯无人机约为 0.4kg，挂载 Delta 后更重）。
  - **UAV 与 Delta 模型的 PID 刚度对比总结**：由于挂载 Delta 机械臂后整个飞行器系统的总质量和转动惯量发生了巨大的改变（特别是机械臂挥动带来的耦合扰动），原先针对轻量级模型的软 PID 参数会导致严重飘逸。
  - 最终确立的强鲁棒性 PID 参数 (`config.py`)：
    - **位置与速度环 (外环)**：放缓起飞瞬间的过冲，但增强了积分项（`VEL_KI=[0.5, 0.5, 1.0]`）以对抗重载情况下的稳态下坠。`POS_KP` 提升至 `2.0`，`VEL_KP` 提升至 `[2.0, 2.0, 3.0]`。
    - **姿态与角速度环 (内环)**：必须大幅提高刚度来压制机械臂的扰动力矩。`ATT_KP` 提升至 `[10.0, 10.0, 6.0]`，`RATE_KP` 暴增至 `[0.15, 0.15, 0.05]`。这套硬参数能保证无人机即使在下挂重物剧烈甩动时也能维持极高的水平稳定性。

---

*注：本次更新标志着 UAV-Delta 仿真核心控制链路的 PyTorch 化重构基本完成，并实现了对无 GUI 远程开发服务器的全面兼容。*
