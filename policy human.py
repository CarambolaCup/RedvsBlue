from ray.rllib.policy.policy import Policy
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg # mpimg 用于读取图片
from IPython.display import clear_output


def show_fig(state):
    width = height = 9
    vlines = np.linspace(-0.5, -0.5+width, width+1)
    hlines = np.linspace(-0.5, -0.5+height, height+1)
    plt.hlines(hlines, -0.5, -0.5+width)
    plt.vlines(vlines, -0.5, -0.5+height)
    plt.axis('off')
    block_pos = np.nonzero(state[:, :, 2])
    blue_pos = np.nonzero(state[:, :, 1])
    red_pos = np.nonzero(state[:, :, 0])
    for i in range(len(block_pos[0])):
        y = block_pos[0][i]
        x = block_pos[1][i]
        plt.fill([x+0.5, x+0.5, x-0.5, x-0.5],
                 [y-0.5, y+0.5, y+0.5, y-0.5], 'black')
    for i in range(len(blue_pos[0])):
        y = blue_pos[0][i]
        x = blue_pos[1][i]
        plt.fill([x+0.5, x+0.5, x-0.5, x-0.5],
                 [y-0.5, y+0.5, y+0.5, y-0.5], 'blue')
    for i in range(len(red_pos[0])):
        y = red_pos[0][i]
        x = red_pos[1][i]
        plt.fill([x+0.5, x+0.5, x-0.5, x-0.5],
                 [y-0.5, y+0.5, y+0.5, y-0.5], 'red')
    plt.axis('equal')
#   plt.savefig('a.png',format='png',dpi=600)
#   a = mpimg.imread('./a.png')
#   plt.imshow(a)
    plt.show()
    plt.close()


class RandomPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_actions(self, obs_batch, *args, **kwargs):
        return [np.random.choice(np.flatnonzero(obs)) for obs in obs_batch['action_mask']], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


class FirstValidPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_actions(self, obs_batch, *args, **kwargs):
        return [np.argmax(obs) for obs in obs_batch['action_mask']], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


class HumanPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_actions(self, obs_batch, *args, **kwargs):
        # player_name = list(obs.keys())[0]
        # assert len(list(obs.keys())) == 1, 'obs中的玩家数多于一个'
        # obs = obs[player_name]
        act_batch=[]
        for obs in obs_batch:
            while(1):
                show_fig(obs['observation'])
                row, col, direction = map(int, input(
                    "请输入行、列(0~8)和方向(0~3)(上下左右)，空格隔开：").split())
                if not (row in range(9) and col in range(9) and direction in range(4)):
                    # os.system('cls') # 执行cls命令清空Python控制台
                    clear_output()  # 清空Notebook代码块输出
                    print('请正确输入行/列/方向！！！')
                    continue
                if obs['action_mask'].reshape(9, 9, 4)[row][col][direction] == 0:
                    # os.system('cls') # 执行cls命令清空Python控制台
                    clear_output()  # 清空Notebook代码块输出
                    print('非法动作！！！')
                    continue
                break
            action = np.zeros((9, 9, 4), dtype=np.int8)
            action[row][col][direction] = 1
            act_num = np.argmax(action.reshape(324))
            act_batch.append(act_num)
        # return {player_name: act_num}
        return act_batch,[],{}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
