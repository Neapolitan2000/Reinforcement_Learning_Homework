from env import Maze
from RL_MC import MCLearningBasic
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(121)
np.random.seed(191)
env = Maze()
MC = MCLearningBasic(actions=list(range(env.n_actions)))
rewards_table = []
episode_num = 20  # 大循环次数
for episode in range(episode_num):
    state = env.reset()  # 初始化环境，得到初始状态
    step_counter = 0  # 每个episode的计数器
    rewards_show = 0  # 回报绘图用
    one_episode_state = []
    while True:
        # 更新可视化环境
        # env.render()

        # 记录一次episode的state
        one_episode_state.append(state)

        # 选择动作
        action = MC.choose_action(str(state))

        # 环境根据动作给出下一个状态，奖励，是否终止
        state_, reward, done, success = env.step(action)

        # 可视化reward累加
        rewards_show += reward

        # 传入下一个状态
        state = state_

        # 计数器
        step_counter += 1

        # 如果终止, 就跳出循环
        if done or step_counter >= 50:
            rewards_table.append(rewards_show)
            interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
            print('\r{}'.format(interaction))
            if success:
                print("success")
            break
    # 用该次episode的state训练
    for i in range(len(one_episode_state)):
        MC.learn(one_episode_state[i])

# end of game
print('game over')
env.show_A_table(MC.A_table)  # 在maze的每个格子中显示最优动作的箭头
env.mainloop()
plt.plot(rewards_table)  # 出图
plt.xlabel('index')
plt.ylabel('reward')
plt.title('Deterministic Maze by MC-learning-basic')
plt.show()
# np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
# MC.show_table()