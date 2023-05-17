import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from env import Maze
from RL_MC import QLearningTable

#mpl.rcParams["font.sans-serif"] = ["SimHei"] #matplot中文字体

def update():
    rewards_table = []
    for episode in range(200):
        # 初始化环境，得到初始状态
        state = env.reset()
        step_counter = 0
        rewards_show = 0
        while True:
            # fresh env
            env.render()

            # RL choose action based on state
            action = RL.choose_action(str(state))

            # RL take action and get next state and reward
            state_, reward, done, success = env.step(action)

            #sum reward to rewards_show
            rewards_show += reward

            # RL learn from this transition
            RL.learn(str(state), action, reward, str(state_))

            # swap state
            state = state_

            # step_counter
            step_counter += 1

            # break while loop when end of this episode
            if done:
                rewards_table.append(rewards_show)
                interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
                print('\r{}'.format(interaction))
                if success:
                    print("success")
                break

    # end of game
    print('game over')
    plt.plot(rewards_table)
    plt.xlabel('index')
    plt.ylabel('reward')
    plt.title('Maze by Q-learning')
    plt.show()
    #np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    RL.show_table()
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()