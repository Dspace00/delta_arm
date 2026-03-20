import torch
import numpy as np
from uav_project.utils.DeltaKinematics import DeltaKinematics

kin = DeltaKinematics()

print("--- Testing IK -> FK Loop ---")
# 1. Target Position in Delta Base Frame (where is this defined? uav or delta frame?)
# The controller gives trajectory:
# x = 0.12 * cos(wt), y = 0.12 * sin(wt), z = -0.18
target_pos = torch.tensor([0.12, 0.0, -0.18], dtype=torch.float32)
print(f"Target Pos: {target_pos.numpy()}")

# 2. Inverse Kinematics
angles_deg = kin.ik(target_pos)
print(f"IK Angles (deg): {angles_deg.numpy()}")

# 3. Forward Kinematics
# FK expects angles in deg
fk_pos = kin.fk(angles_deg)
print(f"FK Pos (from IK angles): {fk_pos.numpy()}")

# Wait, DeltaKinematics has a Frame Transform inside IK!
# mat = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
# It rotates the input by 90 degrees! 
# But FK does NOT have this inverse transform!

