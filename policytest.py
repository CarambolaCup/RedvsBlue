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
    return ATK+DEF


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
    action = action_mask.copy()
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

act = [ [[0,1,0,1],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,1,1,1],[0,0,0,0],[0,1,1,0]],
        [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
        [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
        [[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0]],
        [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
        [[1,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,0]],
        [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
        [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
        [[1,0,0,1],[0,0,0,0],[1,0,1,1],[0,0,0,0],[1,0,1,1],[0,0,0,0],[1,0,1,1],[0,0,0,0],[1,0,1,0]]]

obs = [ [[1,0,0],[0,0,0],[0,1,0],[0,0,0],[1,0,0],[0,0,0],[0,1,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
        [[0,1,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
        [[1,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
        [[0,1,0],[0,0,0],[1,0,0],[0,0,0],[0,1,0],[0,0,0],[1,0,0],[0,0,0],[0,1,0]]]

print(obs2act(act,obs))