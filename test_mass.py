import os
import mujoco
from uav_project.models.uav_model import UAVModel

current_dir = os.path.dirname(os.path.abspath("uav_project/main.py"))
model_path = os.path.join(current_dir, "meshes", "Delta.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

uav = UAVModel(model, data)
print(f"UAV Subtree Mass: {uav.get_mass()}")
print(f"Total Model Mass: {sum(model.body_mass)}")
