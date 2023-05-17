from env import Maze
from RL_MC import MCLearning
import numpy as np
import matplotlib.pyplot as plt


env = Maze()
MC = MCLearning(actions=list(range(env.n_actions)))
rewards_table = []

for episode in range(900):
    print("episode: ", episode)
    if episode == 10:
        state = env.reset_in_state([85, 165, 115, 195])
        MC.learn(state)
        print('10episode', MC.A_table)
    elif episode == 11:
        state = env.reset_in_state([125, 165, 155, 195])
        MC.learn(state)
        print('11episode', MC.A_table)
    else:
        # 初始化环境，得到初始状态
        state = env.reset()
        # 蒙特卡洛训练，针对当前state穷举所有action的价值，比较得出q(s,a)最大的a
        MC.learn(state)
        if episode == 12:
            print('12episode', MC.A_table)





MC.show_table()
env.destroy()