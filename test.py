import numpy as np
import random
from IPython.display import display
state = None
state = np.zeros((9, 9, 3), dtype=np.int8)
board_size = 8
block = 2
cur_player = 0
player_init_place = np.array([[0, 0], [0, 4], [0, 8], [3, 8], [
                             3, 0], [8, 2], [8, 6]])  # 棋子初始位置
for i in range(0, 7):  # 把两种颜色的棋子放到棋盘上
    state[player_init_place[i][0]][player_init_place[i][1]][0] = 1
    state[board_size - player_init_place[i][0]
          ][board_size - player_init_place[i][1]][1] = 1
rand_key = random.sample(range(0, 24), 2)
for i in range(0, 2):
    state[1+rand_key[i] // 7][1+rand_key[i] % 7][block] = 1
    state[board_size-1-rand_key[i] //
          7][board_size-1-rand_key[i] % 7][block] = 1
print(state[:,:,0],state[:,:,1],state[:,:,2],sep='\n\n')
state1=state[:, :, [1,0,2]].copy()
# print(state1[:,:,0],state1[:,:,1],state1[:,:,2],sep='\n\n')
display(state.sum(-1, dtype=np.int8))

def obs(players):
    return {players[player]: {
        'observation': state.copy() if player == 0 else state[:, :, [1,0,2]].copy(),
        'action_mask': 1 - state.sum(-1, dtype=np.int8).flatten()
    } for player in players}

o = obs((0,1))
display(o)
pass
