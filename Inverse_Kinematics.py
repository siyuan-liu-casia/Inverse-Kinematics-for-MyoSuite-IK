import gym
import myosuite
import numpy as np
from scipy.optimize import minimize
'''
A method for calculating joint angles through site point trajectories in a MyoSuite environment. 
In this code, the trajectory of the site is simply interpolated between the starting point and the target point. 
Alternatively, the interpolated_points can be directly assigned as the site trajectory for calculation.
'''
#choose env
env = gym.make('myoElbowPose2D6MFixed-v0')
env.reset()

#set the site start and final position
target_pos_start = np.array([-0.2006, -0.1742, 1.09])
target_pos_final = np.array([-0.2486, -0.3719,  1.1091])

#target site id
wrist_sid = env.sim.model.site_name2id("wrist")

#Simple interpolated trajectory.
interpolated_points = np.linspace(target_pos_start, target_pos_final, 100)

def objective(qpos, target_pos):
    env.sim.data.qpos[:2] = qpos
    env.sim.forward()
    wrist_pos = env.sim.data.site_xpos[wrist_sid]
    return np.sum((wrist_pos - target_pos) ** 2)

initial_guess = env.sim.data.qpos[:2].copy()

all_qpos = []

for target_pos in interpolated_points:
    result = minimize(objective, initial_guess, args=(target_pos,), method='BFGS')
    if result.success:
        optimized_qpos = result.x
        all_qpos.append(optimized_qpos)
        initial_guess = optimized_qpos  # use last result as initial guess 
    else:
        print("Optimization failed for target position:", target_pos)
        all_qpos.append(None)

for idx, qpos in enumerate(all_qpos):
    if qpos is not None:
        print(f"Point {idx}: qpos = {qpos}")
    else:
        print(f"Point {idx}: Optimization failed")

# env.mj_render()
for qpos in all_qpos:
    if qpos is not None:
        env.sim.data.qpos[:2] = qpos
        env.sim.forward()
        env.mj_render()
    else:
        print("Skipping a failed optimization result")

env.close()
