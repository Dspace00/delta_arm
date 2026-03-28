# Delta Robot Simulation Project

A MuJoCo-based simulation platform for Delta robot control, featuring PS2 joystick teleoperation and ball interception tasks.

## Features

- **Delta Robot Simulation**: Independent simulation with base mounted at 4.0m height
- **PS2 Joystick Control**: Real-time teleoperation using 2.4G wireless gamepad
- **Ball Interception**: Predictive trajectory planning and catching system
- **Workspace Analysis**: Automated workspace computation and visualization

---

## Project Structure

```
uav_project/
├── config.py                      # Global configuration parameters
├── config_workspace.py            # Workspace bounds and base height
├── compute_workspace.py           # Workspace analysis tool
│
├── main_delta.py                  # Delta trajectory simulation entry
├── main_delta_joystick.py         # PS2 joystick control entry
├── main_delta_intercept_optimized.py  # Ball interception entry
│
├── hardware/
│   ├── __init__.py
│   └── ps2_controller.py          # PS2 joystick driver
│
├── controllers/
│   ├── delta_arm_controller.py    # Delta base controller
│   ├── delta_intercept_controller_optimized.py  # Interception controller
│   └── pid.py                     # PID implementations
│
├── models/
│   ├── delta_model.py             # Delta sensor/actuator interface
│   └── delta_ball_model.py        # Ball model extension
│
├── utils/
│   ├── DeltaKinematics.py         # Forward/Inverse kinematics
│   ├── ball_predictor.py          # 3-point trajectory fitting
│   ├── ball_trajectory_generator.py   # Collision-free trajectory
│   ├── smooth_trajectory.py       # C4-continuous trajectory planning
│   └── logger.py                  # Data recording and visualization
│
└── meshes/
    ├── Delta_Arm.xml              # Delta robot model
    └── Delta_Ball.xml             # Delta + Ball model
```

---

## Installation

### Prerequisites

- Python 3.11
- Conda (recommended)

### Environment Setup

```bash
# Create conda environment
conda env create -f environment_win.yml

# Activate environment
conda activate mujoco-sim-win

# Verify installation
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## Usage

### 1. PS2 Joystick Teleoperation

Control the Delta robot using a PS2 2.4G wireless gamepad.

**Hardware Requirements:**
- PS2 2.4G wireless gamepad with USB receiver
- System detects as "USB WirelessGamepad" (HID device)

**Running:**
```bash
cd Project_uav
python -m uav_project.main_delta_joystick
```

**Controls:**
| Button/Axis | Function |
|-------------|----------|
| Left Stick X/Y | End-effector X/Y position |
| Left Stick Up/Down | End-effector Z position (hold L2) |
| L2 + Left Stick | Z-axis control |
| R2 | Reset to home position |

**Files:**
- `hardware/ps2_controller.py`: Joystick driver implementation
- `main_delta_joystick.py`: Main entry point

### 2. Ball Interception Task

The Delta robot intercepts and catches flying balls.

**Running:**
```bash
cd Project_uav
python -m uav_project.main_delta_intercept_optimized
```

**Key Features:**
- **3-Point Trajectory Fitting**: Uses 2 points for fitting, 1 for validation
- **Virtual Paddle Model**: 7.5cm radius paddle centered 10cm above platform
- **Collision Avoidance**: Trajectories avoid Delta base and arm links
- **C4-Continuous Motion**: Smooth trajectory with continuous derivatives

**Configuration:**
- Base height: 4.0m (configurable in `config_workspace.py`)
- Workspace bounds: Auto-computed from kinematics
- Catch threshold: 9.5cm (paddle radius + ball radius)

**Files:**
- `main_delta_intercept_optimized.py`: Main entry point
- `controllers/delta_intercept_controller_optimized.py`: Interception logic
- `utils/ball_predictor.py`: Trajectory prediction
- `utils/ball_trajectory_generator.py`: Trajectory generation

### 3. Trajectory Simulation

Run Delta robot along predefined trajectories.

```bash
cd Project_uav
python -m uav_project.main_delta
```

---

## Configuration

### Workspace Bounds (`config_workspace.py`)

```python
# Base height in world frame
BASE_HEIGHT = 4.0  # meters

# Workspace bounds (relative to base)
WORKSPACE_BOUNDS = {
    'x': (-0.07, 0.07),   # ±7cm
    'y': (-0.08, 0.05),   # -8cm to +5cm
    'z': (-0.19, -0.09)   # Below base
}

# Effective XY radius
WORKSPACE_RADIUS = 0.055  # meters
```

### Coordinate System

```
World Frame:          Local Frame (relative to base):
      Z ↑                    Z ↑
        |                      |
        |______→ Y             |______→ Y
       /                      /
      /                      /
     ↙ X                    ↙ X

Base at z = 4.0m          Base at z = 0.0m
```

**Conversion:**
```python
# World → Local
local_z = world_z - BASE_HEIGHT

# Local → World
world_z = local_z + BASE_HEIGHT
```

---

## Key Algorithms

### 1. Delta Kinematics

- **Forward Kinematics**: Joint angles → End-effector position
- **Inverse Kinematics**: End-effector position → Joint angles
- Implemented in `utils/DeltaKinematics.py`

### 2. 3-Point Trajectory Fitting

Physics model: `r(t) = r₀ + v₀·t + 0.5·g·t²`

- Collect 3 position samples from ball trajectory
- Use 2 points to solve for initial position and velocity
- Use 3rd point for validation (threshold: 10cm)

### 3. Virtual Paddle Catch Model

```
        ┌─────────────┐
        │   Paddle    │  ← 7.5cm radius
        │  (virtual)  │
        └──────┬──────┘
               │
          10cm │ sensor offset
               │
        ┌──────┴──────┐
        │  Platform   │  ← Physical platform
        └─────────────┘
```

**Catch Condition:** `distance(ball_center, paddle_center) < 9.5cm`

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mujoco | ≥3.0.0 | Physics simulation |
| torch | 2.3.1 | Controller computation |
| numpy | <2.0 | MuJoCo data interface |
| scipy | - | Rotation/quaternion |
| matplotlib | - | Visualization |
| pygame | ≥2.5.0 | Joystick input |

---

## Results

### Simulation Screenshots

Located in `uav_project/`:
- `delta_workspace_3d.png`: 3D workspace visualization
- `joystick_simulation_results.png`: Joystick control results
- `optimized_interception_results.png`: Interception task results

### Example Outcomes

- `optimized_caught_example.png`: Successful catch
- `optimized_missed_example.png`: Missed catch

---

## Known Issues

1. **Coordinate System Inconsistency**: The ball interception system currently has low catch rate due to mixed use of world and local coordinates. This is being addressed.

2. **Trajectory Re-planning**: Temporarily disabled to avoid timing issues.

---

## Development

### Adding New Controllers

1. Create controller in `controllers/`
2. Inherit from `DeltaArmController`
3. Implement required methods:
   - `update(sim_time)`
   - `get_log_data()`
   - `set_target_position(pos)`

### Data Type Convention

```
MuJoCo API ←→ NumPy (float64)
     ↕
Controller  ←→ PyTorch Tensor (float32)
```

---

## License

MIT License

---

## Author

Delta Robot Simulation Project Team
