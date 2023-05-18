import numpy as np
import pandas as pd
import random
from env import Maze

def moving_average(data, window_size):  # 计算滑动平均值
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    smoothed_data = cumsum[window_size - 1:] / window_size
    return smoothed_data

class MCLearningBasic:
    def __init__(self, actions, gamma=0.98, epsilon=1):
        self.actions = actions  # list=[0, 1, 2, 3, 4]
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # 初始化q表
        self.A_table = pd.DataFrame(dtype=int)  # 初始化A表
        self.env_in = Maze()  # 初始化环境

    def choose_action(self, state):
        self.check_action_exist(str(state))  #  检查是否有该状态的行
        self.check_state_exist(str(state))  # 检查是否有该状态的行
        # 从A表中获取该状态的行，遵从epislon-greedy策略选择动作
        if random.random() < self.epsilon:
            # 从A表中获取该状态的行
            state_action = self.A_table.loc[str(state)]
            # 从该状态的行中获取动作
            action = state_action.item()
        else:
            # 随机选择动作
            action = np.random.choice(self.actions)
        return action

    def learn(self, state_out):
        self.check_action_exist(str(state_out))  # 检查是否有该状态的行
        self.check_state_exist(str(state_out))  # 检查是否有该状态的行
        for action_out in self.actions:  # 对每一个动作遍历
            self.G_average = 0  # 初始化平均回报G
            for sample_num in range(20):
                self.env_in.reset_in_state(state_out)  # 重置环境到指定状态
                a = action_out  # 保存动作action_out
                self.G = 0  # 初始化回报G
                episodes = []  # 记录一条轨迹上的数据对
                count = 0
                s_, r, done, success = self.env_in.step(a)  # 采取动作a，得到下一个状态s_，奖励r，是否终止done，是否成功success
                episodes.append([state_out, action_out, r])  # 记录一条轨迹上的数据对
                s = s_
                while (not done) and (count < 50):  # 如果没有终止，就一直采取动作，直到400步上限
                    # self.env_in.render()  # 更新可视化环境
                    action = self.choose_action(str(s))  # 选择动作
                    s_, r, done, success = self.env_in.step(action)  # 执行一步
                    episodes.append([s, action, r])  # 记录一条轨迹上的数据对
                    s = s_
                    count += 1
                # print(episodes)
                # breakpoint()
                for i in range(len(episodes)-1, -1, -1):  # 从后往前遍历
                    self.G = self.gamma * self.G + episodes[i][2]  # 计算回报
                # 更新q表，递增平均数更新：Q(s,a) = Q(s,a) + 1/N(s,a) * (G - Q(s,a))
                self.G_average = self.G_average + (self.G - self.G_average) / (sample_num + 1)
            self.q_table.loc[str(state_out), action_out] = self.G_average  # 更新q表
        # 策略更新
        state_action = self.q_table.loc[str(state_out), :]  # 获取q表中该状态的行
        # action = np.random.choice(state_action[state_action == np.max(state_action)].index)  # 从该状态的行中获取最大值的动作
        best_action = np.min(state_action[state_action == np.max(state_action)].index)  # 从该状态的行中获取最大值的动作的最小序列
        self.A_table.loc[str(state_out)] = int(best_action)  # 更新A表

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 如果是没有探索过的state，就添加到q表中并赋予一个全0的q值
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def check_action_exist(self, state):
        if state not in self.A_table.index:
            # 如果是没有探索过的state，就添加到A表中并赋予一个随机动作作为初始策略
            self.A_table = self.A_table.append(
                pd.Series(
                    random.randint(0, 4),
                    name=state,
                )
            )


    def show_table(self, ):
        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
        print('\rAtable\r', self.A_table)
        print('\rQtable:\r', self.q_table)
        print(self.q_table.loc[str([85.0, 165.0, 115.0, 195.0]), :])
        print(self.q_table.loc[str([125.0, 165.0, 155.0, 195.0]), :])
        print(self.q_table.loc[str([165.0, 165.0, 195.0, 195.0]), :])

class MCLearningExplore:
    def __init__(self, actions, gamma=0.98, epsilon=1):
        self.actions = actions  # list=[0, 1, 2, 3, 4]
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # 初始化q表
        self.num_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # 初始化num表
        self.A_table = pd.DataFrame(dtype=int)  # 初始化A表

    def choose_action(self, state):
        self.check_action_exist(str(state))  #  检查是否有该状态的行
        self.check_state_exist(str(state))  # 检查是否有该状态的行
        # 从A表中获取该状态的行，遵从epislon-greedy策略选择动作
        if np.random.uniform() < self.epsilon:
            # 从A表中获取该状态的行
            state_action = self.A_table.loc[str(state)]
            # 从该状态的行中获取动作
            action = state_action.item()
        else:
            # 随机选择动作
            action = np.random.choice(self.actions)
        return action

    def learn(self, episodes):

        # 初始化G
        self.G = 0
        for i in range(len(episodes)-1, -1, -1):  # 从后往前遍历
            state = episodes[i][0]  # 获取状态
            action = episodes[i][1]  # 获取动作
            reward = episodes[i][2]  # 获取奖励
            self.check_state_exist(str(state))  # 初始化该状态对应的q表和计数表
            self.check_action_exist(str(state))  # 初始化该状态对应的A表
            self.num_table.loc[str(state), action] += 1  # 该状态动作对的计数加1
            self.G = self.gamma * self.G + reward  # 计算回报
            # 更新q表，递增平均数更新：Q(s,a) = Q(s,a) + 1/N(s,a) * (G - Q(s,a))
            self.q_table.loc[str(state), action] = self.q_table.loc[str(state), action] + 1/self.num_table.loc[str(state), action] * (self.G - self.q_table.loc[str(state), action])
        # 策略更新，对q表遍历状态
        for state in self.q_table.index:
            state_action = self.q_table.loc[str(state), :]  # 获取q表中该状态的行
            best_action = np.min(state_action[state_action == np.max(state_action)].index)  # 从该状态的行中获取最大值的动作的最小序列
            self.A_table.loc[str(state)] = int(best_action)  # 更新A表

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 如果是没有探索过的state，就添加到q表中并赋予一个全0的q值
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        if state not in self.num_table.index:
            # 如果是没有探索过的state，就添加到num表中并赋予一个全0的num值
            self.num_table = self.num_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.num_table.columns,
                    name=state,
                )
            )

    def check_action_exist(self, state):
        if state not in self.A_table.index:
            # 如果是没有探索过的state，就添加到A表中并赋予一个随机动作作为初始策略
            self.A_table = self.A_table.append(
                pd.Series(
                    random.randint(0, 4),
                    name=state,
                )
            )


    def show_table(self, ):
        np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

        print(self.A_table)
        print(self.q_table.loc[str([85.0, 165.0, 115.0, 195.0]), :])
        print(self.q_table.loc[str([125.0, 165.0, 155.0, 195.0]), :])
        print(self.q_table.loc[str([165.0, 165.0, 195.0, 195.0]), :])