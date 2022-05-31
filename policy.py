# 这是一个写policy的示例，只用修改MyPolicy里的compute_actions
# 就是给你一个batch的obs（比如说是一个包含32个obs的列表）
# 然后你返回给它一个batch的action
# 实际上就是你要对每个obs给出一个action。
# 只是系统会一次性给你batch_size个obs，然后你也一次性给它返回batch_size个action
# from audioop import reverse
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


ALLY = 0
ENEMY = 1
WALL = 2

def player_situation_detect(x, y, observation):
    ATK = 10
    DEF = 0
    cpy_ally = 0
    cpy_enemy = 0
    for offset_x in range(-3, +3):
        for offset_y in range(-3, +3):
            if 0 <= x+offset_x and x+offset_x <= 8 and 0 <= y+offset_y and y+offset_y <= 8:
                if not (0 == offset_x and 0 == offset_y):
                    if 1 == observation[x+offset_x][y+offset_y][ALLY]:
                        cpy_ally = cpy_ally+1
                        ATK = ATK + 100 // (abs(offset_x)+abs(offset_y))
                        DEF = DEF + 100 // (abs(offset_x)+abs(offset_y))
                    elif 1 == observation[x+offset_x][y+offset_y][ENEMY]:
                        cpy_enemy = cpy_enemy + 1
                        DEF = DEF - 110 // (abs(offset_x)+abs(offset_y))
                    elif 1 == observation[x+offset_x][y+offset_y][WALL]:
                        ATK = ATK - 20 // (abs(offset_x)+abs(offset_y))
                        DEF = DEF + 30 // (abs(offset_x)+abs(offset_y))
    for offset_x in range(-3, +3):
        for offset_y in range(-3, +3):
            if not (0 <= x+offset_x and x+offset_x <= 8 and 0 <= y+offset_y and y+offset_y <= 8):
                if cpy_ally > cpy_enemy:
                    ATK = ATK + 25
                elif cpy_ally < cpy_enemy:
                    DEF = DEF - 20
    return (ATK, DEF)


def move_pririty(ATK, DEF):
    return ATK+DEF / 2


def closest(x, y, observation):
    ally_n = 0
    enemy_n = 0
    ally_x = 0
    ally_y = 0
    enemy_x = 0
    enemy_y = 0
    for offset_x in range(-3, +3):
        for offset_y in range(-3, +3):
            if 0 <= x+offset_x and x+offset_x <= 8 and 0 <= y+offset_y and y+offset_y <= 8:
                if not (0 == offset_x and 0 == offset_y):
                    if 1 == observation[x+offset_x][y+offset_y][ALLY]:
                        ally_n = ally_n+1
                        ally_x = ally_x*(ally_n-1) / ally_n
                        ally_y = ally_y*(ally_n-1) / ally_n
                    elif 1 == observation[x+offset_x][y+offset_y][ENEMY]:
                        enemy_n = enemy_n+1
                        enemy_x = enemy_x*(enemy_n-1) / enemy_n
                        enemy_y = enemy_y*(enemy_n-1) / enemy_n
    if 0 == ally_n:
        ally_x = x
        ally_y = y
    if 0 == enemy_n:
        enemy_x = x
        enemy_y = y
    return ally_x, ally_y, enemy_x, enemy_y

def obs2act(action_mask, observation):
    def move324(i, j, k):
        return i*36 + j * 4 + k
    pre_move = -1
    action = action_mask.reshape(9,9,4)
    d0 = np.array([-1, 1, 0, 0], dtype=np.int8)
    d1 = np.array([0, 0, -1, 1], dtype=np.int8)

    pos = []
    for x in range(9):
        for y in range(9):
            if observation[x][y][ALLY]:
                pos.append((x,y))
    attitude = [player_situation_detect(
        x, y, observation) for x, y in pos]

    move_p = [move_pririty(a, d) for a, d in attitude]
    move_queue = np.argsort(np.abs(move_p))[::-1]
    for i in move_queue:
        if np.any(action[pos[move_queue[i]][0]][pos[move_queue[i]][1]]):
            poss_act = np.nonzero(action[pos[move_queue[i]][0]][pos[move_queue[i]][1]])
            a_x, a_y, e_x, e_y = closest(
                pos[move_queue[i]][0], pos[move_queue[i]][1], observation)
            for j in range(4):
                if 1 == action[pos[move_queue[i]][0]][pos[move_queue[i]][1]][j]:
                    if -1 == pre_move:
                        pre_move = move324(
                            pos[move_queue[i]][0], pos[move_queue[i]][1], j)
                    if move_p[i] > 0:
                        if d0[j]*(e_x-pos[move_queue[i]][0])+d1[j]*(e_y - pos[move_queue[i]][1]) > 0:
                            pre_move = move324(
                                pos[move_queue[i]][0], pos[move_queue[i]][1], j)
                            return pre_move
                    else:
                        if d0[j]*(a_x-pos[move_queue[i]][0])+d1[j]*(a_y - pos[move_queue[i]][1]) > 0:
                            pre_move = move324(
                                pos[move_queue[i]][0], pos[move_queue[i]][1], j)
                            return pre_move
    return pre_move


class MyPolicy(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_actions(self, obs_batch, *args, **kwargs):
        return [obs2act(obs_mask, obs_obsv) for obs_mask, obs_obsv in zip(obs_batch["action_mask"], obs_batch["observation"])], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


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
        act_batch = []
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
        return act_batch, [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
