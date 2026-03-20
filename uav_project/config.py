"""
Configuration parameters for the UAV simulation and control.
"""
import numpy as np

# --- Simulation Settings ---
# Note: The actual simulation timestep is defined in the XML file (typically 0.0001s).
# This variable is for reference or if we want to override/check it.
SIM_TIMESTEP = 0.0001  
RENDER_FPS = 60

# --- UAV Physical Parameters ---
MASS = 0.4  # kg (Aligned with UAV.xml / Delta.xml UAV_body mass)
GRAVITY = 9.81  # m/s^2
ARM_LENGTH = 0.18  # m (Motor arm length) the distance from the axis to the motor, which means the total length is equal to sqrt(2) * ARM_LENGTH

#                x-aixs
#               ^
# (1)           |            (2)
#    \\\\       |       ////
#       \\\\    |    ////
# y-axis   \\\\ | ////
# <------------(+)-------------
#          //// | \\\\
#       ////    |    \\\\
#    ////       |       \\\\
# (4)           |            (3)

# prop1/prop3: anti_clockwise (CCW)
# prop2/prop4: clockwise (CW)

# --- Motor Parameters ---
# # Max thrust per motor (N)
# MAX_THRUST_PER_MOTOR = 6.5
# # Max torque per motor (Nm)
# MAX_TORQUE_PER_MOTOR = 0.15
# # Max motor speed (krpm)
MAX_MOTOR_SPEED_KRPM = 22.0

# Thrust coefficient (N / krpm^2) 
CT = 0.01343
# Torque coefficient (Nm / krpm^2)
CM = 0.00031



# --- Controller Frequencies (Hz) ---
FREQ_DELTA = 100
FREQ_POSITION = 50
FREQ_VELOCITY = 50
FREQ_ATTITUDE = 250
FREQ_ANGLE_RATE = 1000

# --- PID Gains ---
# Position Control (P-only)
POS_KP = np.array([2.0, 2.0, 2.0])

# Velocity Control (PID)
VEL_KP = np.array([2.0, 2.0, 3.0])
VEL_KI = np.array([0.5, 0.5, 1.0])
VEL_KD = np.array([0.05, 0.05, 0.05])

# Attitude Control (P-only, Quaternion based)
ATT_KP = np.array([10.0, 10.0, 6.0])

# Angular Rate Control (PID)
RATE_KP = np.array([0.15, 0.15, 0.05])
RATE_KI = np.array([0.05, 0.05, 0.02])
RATE_KD = np.array([0.005, 0.005, 0.001])

# --- Output Limits ---
# Can be added here if needed, currently handled in logic
