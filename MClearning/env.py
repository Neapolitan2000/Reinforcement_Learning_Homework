import numpy as np
import time
import tkinter as tk
import random


UNIT = 40   # pixels(40,4,4)
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r', 's']  # 上up下down左left右right和不动stay
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # 划线
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 初始坐标点
        origin = np.array([20, 20])

        # 禁止区域hell
        hell1_center = origin + np.array([UNIT * 1, UNIT * 1])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='orange')
        hell2_center = origin + np.array([UNIT * 2, UNIT * 1])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='orange')
        hell3_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='orange')
        hell4_center = origin + np.array([UNIT, UNIT * 3])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='orange')
        hell5_center = origin + np.array([UNIT, UNIT * 4])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='orange')
        hell6_center = origin + np.array([UNIT * 3, UNIT * 3])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='orange')

        # 目标区域
        oval_center = origin + np.array([UNIT * 2, UNIT * 3])
        self.oval = self.canvas.create_rectangle(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='blue')

        #风场区域
        # wind_center1 = origin + np.array([UNIT * 3, UNIT * 1])
        # self.wind1 = self.canvas.create_rectangle(
        #     wind_center1[0] - 15, wind_center1[1] - 15,
        #     wind_center1[0] + 15, wind_center1[1] + 15,
        #     fill='green')
        # wind_center2 = origin + np.array([UNIT * 4, UNIT * 1])
        # self.wind2 = self.canvas.create_rectangle(
        #     wind_center2[0] - 15, wind_center2[1] - 15,
        #     wind_center2[0] + 15, wind_center2[1] + 15,
        #     fill='green')
        # wind_center3 = origin + np.array([UNIT * 4, UNIT * 2])
        # self.wind3 = self.canvas.create_rectangle(
        #     wind_center3[0] - 15, wind_center3[1] - 15,
        #     wind_center3[0] + 15, wind_center3[1] + 15,
        #     fill='green')
        # wind_center4 = origin + np.array([UNIT * 3, UNIT * 2])
        # self.wind4 = self.canvas.create_rectangle(
        #     wind_center4[0] - 15, wind_center4[1] - 15,
        #     wind_center4[0] + 15, wind_center4[1] + 15,
        #     fill='green')

        # 运动体rect
        self.rect = self.canvas.create_oval(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()  # 更新画布
        time.sleep(0.5)  # 展示用，可以缩短时间训练
        self.canvas.delete(self.rect)  # 删除原来的rect
        # origin = np.array([20, 20])  # reset出发点为左上角初始点
        origin = np.array([20 + np.random.randint(0, MAZE_H - 1) * UNIT, 20 + np.random.randint(0, MAZE_W - 1) * UNIT])  # reset出发点为任意一点
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # 返回状态
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)  # 获取当前位置(状态)
        hit_wall = False  # 检测碰壁情况
        base_action = np.array([0, 0])
        '''
        if s in [self.canvas.coords(self.wind1), self.canvas.coords(self.wind2),
                 self.canvas.coords(self.wind3), self.canvas.coords(self.wind4),]:  # 如果s在风场区域
            base_action, hit_wall = self.move_stochastic(action, s, base_action, hit_wall)
        else:  # 如果在正常区域
            base_action, hit_wall = self.move_deterministic(action, s, base_action, hit_wall)
        '''  # 风场情况

        base_action, hit_wall = self.move_deterministic(action, s, base_action, hit_wall)  # 无风场情况
        self.canvas.move(self.rect, base_action[0], base_action[1])  # 移动agent

        s_ = self.canvas.coords(self.rect)  # 下一个状态

        # reward设置
        if s_ == self.canvas.coords(self.oval):
            reward = 1  # 到达目标区域奖励为1
            done = True
            success = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3),
                    self.canvas.coords(self.hell4), self.canvas.coords(self.hell5), self.canvas.coords(self.hell6)]:
            reward = -1  # 掉入陷阱奖励为-1
            done = True
            success = False
            s_ = 'terminal'
        # elif s in [self.canvas.coords(self.wind1), self.canvas.coords(self.wind2),
        #             self.canvas.coords(self.wind3), self.canvas.coords(self.wind4),] and hit_wall:  # 如果s在风场区域向右撞墙
        #     if random.random < 0.2:
        #         reward = 0
        #     else:
        #         reward = -1
        #     done = False
        #     success = False
        elif hit_wall:
            reward = -1  # 碰壁奖励为-1
            done = False
            success = False
        else:
            reward = 0  # 其他情况奖励为0
            done = False
            success = False

        return s_, reward, done, success

    def move_deterministic(self, action, s, base_action, hit_wall):
        if action == 0:   # 向上
            if s[1] > UNIT:
                base_action[1] -= UNIT
            else:
                hit_wall = True  # 碰壁
        elif action == 1:   # 向下
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
            else:
                hit_wall = True  # 碰壁
        elif action == 2:   # 向左
            if s[0] > UNIT:
                base_action[0] -= UNIT
            else:
                hit_wall = True  # 碰壁
        elif action == 3:   # 向右
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
            else:
                hit_wall = True  # 碰壁
        elif action == 4:   # 不动
            base_action[0] += 0
        return base_action, hit_wall

    def move_stochastic(self, action, s, base_action, hit_wall):
        if action == 0:   # 采取向上的动作
            if random.random() < 0.2:  # 按20%几率向左移一格
                base_action[0] -= UNIT  # 向左移动一格
            elif s[1] > UNIT:
                base_action[1] -= UNIT  # 向上移动一格
            else:
                hit_wall = True  # 碰壁并保持不动
        elif action == 1:   # 采取向下动作
            if random.random() < 0.2:  # 按20%几率向左移一格
                base_action[0] -= UNIT  # 向左移动一格
            elif s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
            else:
                hit_wall = True  # 碰壁
        elif action == 2:   # 采取向左动作
            if s[0] > UNIT:
                base_action[0] -= UNIT
            else:
                hit_wall = True  # 碰壁
        elif action == 3:   # 采取向右动作
            if random.random() < 0.2:  # 按20%几率向左移一格
                base_action[0] -= UNIT  # 向左移动一格
            elif s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
            else:
                hit_wall = True  # 碰壁
        elif action == 4:   # 采取不动的动作
            if random.random() < 0.2:  # 按20%几率向左移一格
                base_action[0] -= UNIT  # 向左移动一格

        return base_action, hit_wall


    def render(self):
        time.sleep(0.005)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done, success = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()