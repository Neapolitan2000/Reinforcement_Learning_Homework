from env import Maze
from RL_MC import MCLearningBasic
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(111)
np.random.seed(111)
env = Maze()
MC = MCLearningBasic(actions=list(range(env.n_actions)))
rewards_table = []
for episode in range(200):
    if episode == 0:
        for i in range(9):
            if i == 4:
                state = env.reset_in_state([85, 165, 115, 195])
                MC.learn(state)
            elif i == 5:
                state = env.reset_in_state([125, 165, 155, 195])
                MC.learn(state)
            else:
                state = env.reset()
                MC.learn(state)
    # 初始化环境，得到初始状态
    state = env.reset()
    step_counter = 0
    rewards_show = 0
    one_episode_state = []
    while True:
        # 更新可视化环境
        # env.render()

        #记录一次episode的state
        one_episode_state.append(state)

        # 选择动作
        action = MC.choose_action(str(state))

        # 环境根据动作给出下一个状态，奖励，是否终止
        state_, reward, done, success = env.step(action)

        # 可视化reward累加
        rewards_show += reward

        # if not done:
        #     # 蒙特卡洛训练，针对当前state穷举所有action的价值，比较得出q(s,a)最大的a
        #     MC.learn(state)

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
plt.plot(rewards_table)
plt.xlabel('index')
plt.ylabel('reward')
plt.title('Maze by MC-learning-basic')
plt.show()
#np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
MC.show_table()
env.destroy()