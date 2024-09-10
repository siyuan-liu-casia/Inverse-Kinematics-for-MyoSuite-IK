import numpy as np
import mujoco as mj
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
'''
A method for calculating joint angles through site point trajectories in mujoco with myosim xml model. 
In this code, the trajectory of the site is simply interpolated between the starting point and the target point. 
Alternatively, the interpolated_points can be directly assigned as the site trajectory for calculation.
'''
# Mujoco settings
sim_time = 0.5
#your myosim model xml path
xml_path = r'F:\pycode\MyoSuite\myosuite-main\myosuite\simhive\myo_sim\elbow\myoelbow_2dof6muscles.xml'

# Load Mujoco model and data
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Target positions for the site (in this case, 'wrist')
target_pos_start = np.array([-0.2006, -0.1742, 1.09])
target_pos_final = np.array([-0.2486, -0.3719, 1.1091])

# Interpolate the trajectory between start and final positions
interpolated_points = np.linspace(target_pos_start, target_pos_final, 100)

# get target site id
site_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
wrist_sid = None
for i, name in enumerate(site_names):
    if name == 'wrist':
        wrist_sid = i
        break

def objective(qpos, target_pos):
    # Set joint positions in data
    data.qpos[:2] = qpos
    # Forward simulate to update the site position
    mj.mj_forward(model, data)
    # Get the position of the wrist site
    wrist_pos = data.site_xpos[wrist_sid]
    # Calculate the squared distance between the current wrist position and the target
    return np.sum((wrist_pos - target_pos) ** 2)

# Initial guess for the joint positions (qpos)
initial_guess = data.qpos[:2].copy()

# Store all the optimized joint positions
all_qpos = []

# Optimize joint angles to follow the interpolated trajectory
for target_pos in interpolated_points:
    result = minimize(objective, initial_guess, args=(target_pos,), method='BFGS')
    if result.success:
        optimized_qpos = result.x
        all_qpos.append(optimized_qpos)
        initial_guess = optimized_qpos  # Use the last result as the next initial guess
    else:
        print("Optimization failed for target position:", target_pos)
        all_qpos.append(None)

print(all_qpos)
