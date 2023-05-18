from env import Maze
from RL_MC import MCLearningExplore
from RL_MC import moving_average
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(23)
np.random.seed(23)
env = Maze()
MC = MCLearningExplore(actions=list(range(env.n_actions)), gamma=0.98, epsilon=0.9)
rewards_table = []
for episode in range(10000):
    # 初始化环境，得到初始状态
    state = env.reset()
    step_counter = 0
    rewards_show = 0
    one_episode_state = []
    while True:
        # 更新可视化环境
        # env.render()

        if step_counter == 0:
            # 第一次随机选择动作
            action = np.random.choice(list(range(env.n_actions)))
        else:
            # 选择动作
            action = MC.choose_action(str(state))

        # 环境根据动作给出下一个状态，奖励，是否终止
        state_, reward, done, success = env.step(action)

        # 记录一次episode的state
        one_episode_state.append([state, action, reward])

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
    # 用该次episode的数据训练
    MC.learn(one_episode_state)

# end of game
print('game over')
# plt.plot(rewards_table)
# plt.xlabel('index')
# plt.ylabel('reward')
# plt.title('Maze by MC-learning-explore-epsilon')
env.show_A_table(MC.A_table)  # 在maze的每个格子中显示最优动作的箭头
env.mainloop()
#  新建一个图
plt.figure(2)
reward_table_smooth = moving_average(rewards_table, 500)
plt.plot(reward_table_smooth)
plt.xlabel('index')
plt.ylabel('reward_smooth')
plt.title('Maze by MC-learning-explore-epsilon')
plt.show()
#np.set_printoptions(formatter={'float': '{: 0.5f}'.format})