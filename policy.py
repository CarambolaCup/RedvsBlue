# 这是一个写policy的示例，只用修改MyPolicy里的compute_actions
# 就是给你一个batch的obs（比如说是一个包含32个obs的列表）
# 然后你返回给它一个batch的action
# 实际上就是你要对每个obs给出一个action。
# 只是系统会一次性给你batch_size个obs，然后你也一次性给它返回batch_size个action
from audioop import reverse
from ray.rllib.policy.policy import Policy
import numpy as np

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
    return ATK+DEF*3//2


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
    action = action_mask.reshap(9, 9, 4)
    d0 = np.array([-1, 1, 0, 0], dtype=np.int8)
    d1 = np.array([0, 0, -1, 1], dtype=np.int8)

    pos = np.nonzero(observation[ALLY])
    attitude = [player_situation_detect(
        x, y, observation) for x, y in zip(pos[0], pos[1])]

    move_p = [move_pririty(a, d) for a, d in attitude]
    move_queue = reverse(np.argsort(np.abs(move_p)))
    for i in move_queue:
        if np.any(action[pos[0][move_p[i]]][pos[1][move_p[i]]]):
            poss_act = np.nonzero(action[pos[0][move_p[i]]][pos[1][move_p[i]]])
            a_x, a_y, e_x, e_y = closest(
                pos[0][move_p[i]], pos[1][move_p[i]], observation)
            for j in range(4):
                if 1 == action[pos[0][move_p[i]]][pos[1][move_p[i]]][j]:
                    if -1 == pre_move:
                        pre_move = move324(pos[0][move_p[i]], pos[1][move_p[i]], j)
                    if move_p[i] > 0:
                        if d0[j]*(e_x-pos[0][move_p[i]])+d1[j]*(e_y - pos[1][move_p[i]]) > 0:
                            pre_move = move324(pos[0][move_p[i]], pos[1][move_p[i]], j)
                            return pre_move
                    else:
                        if d0[j]*(a_x-pos[0][move_p[i]])+d1[j]*(a_y - pos[1][move_p[i]]) > 0:
                            pre_move = move324(pos[0][move_p[i]], pos[1][move_p[i]], j)
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
