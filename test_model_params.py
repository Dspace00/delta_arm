import mujoco
from uav_project.models.uav_model import UAVModel

# Load the XML
model = mujoco.MjModel.from_xml_path('uav_project/meshes/UAV.xml')
data = mujoco.MjData(model)

# Original gear ratios from XML
print("Original Gear Ratios from XML:")
for i in range(4):
    actuator_name = f'rotor{i}'
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    print(f"{actuator_name}: {model.actuator_gear[actuator_id]}")

# Initialize UAVModel (which should dynamically update the gear ratios)
uav = UAVModel(model, data)

# Updated gear ratios
print("\nUpdated Gear Ratios after UAVModel init:")
for i in range(4):
    actuator_name = f'rotor{i}'
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    print(f"{actuator_name}: {model.actuator_gear[actuator_id]}")

