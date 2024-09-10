import gym
import myosuite
import numpy as np
from scipy.optimize import minimize

env = gym.make('myoElbowPose2D6MFixed-v0')
env.reset()

target_pos_start = np.array([-0.2006, -0.1742, 1.09])
target_pos_final = np.array([-0.2486, -0.3719,  1.1091])

# 获取名为'wrist'的site点的ID
wrist_sid = env.sim.model.site_name2id("wrist")

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
        initial_guess = optimized_qpos  # 使用上一个结果作为下一个初始猜测
    else:
        print("Optimization failed for target position:", target_pos)
        all_qpos.append(None)

for idx, qpos in enumerate(all_qpos):
    if qpos is not None:
        print(f"Point {idx}: qpos = {qpos}")
    else:
        print(f"Point {idx}: Optimization failed")

# 将all_qpos中的点逐个输入给环境，并利用env.mj_render()显示
for qpos in all_qpos:
    if qpos is not None:
        env.sim.data.qpos[:2] = qpos
        env.sim.forward()
        env.mj_render()
    else:
        print("Skipping a failed optimization result")

env.close()




# wrist_sid = env.sim.model.site_name2id("wrist")

# # 将qpos的两个关节角都设置为0
# # env.sim.data.qpos[:2] = [0,1.5]
# env.sim.data.qpos[:2] = [1, 0]

# env.sim.forward()
# env.mj_render()
# print(env.sim.data.site_xpos[wrist_sid])
# for _ in range(100):
#     env.mj_render()
#     env.step(env.action_space.sample()) # take a random action
#     print(env.sim.data.site_xpos[wrist_sid])
# env.close()
