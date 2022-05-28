from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Box, Discrete, Dict
import numpy as np
import random

class TicTacToeEnv(MultiAgentEnv):
    observation_space = Dict(dict(observation = Box(0, 1, (9, 9, 3), np.int8), action_mask = Box(0, 1, (81,), np.int8)))
    action_space = Discrete(81)
    players = ('player_1', 'player_2')
    
    

    def __init__(self, config):
        self.state = None
        self.board_size=8
        self.players_1 = 0
        self.players_2 = 1
        self.block = 2
    def reset(self):
        self.state = np.zeros((9, 9, 3), dtype = np.int8)
        self.cur_player = 0
        player_ini_place = np.array([[0,0],[0,4],[0,8],[3,8],[5,0],[8,2],[8,6]]) #棋子初始位置
        for i in range(0,7): #把两种颜色的棋子放到棋盘上
            self.state[ player_ini_place[i][0] ][ player_ini_place[i][1] ][0]=1
            self.state[ self.board_size - player_ini_place[i][0] ][ self.board_size - player_ini_place[i][1] ][1]=1
        rand_key=random.sample(range(0,24),2)
        for i in range(0,2):
            self.state[ 1+rand_key[i] // 7 ][ 1+rand_key[i] % 7 ][self.block]=1
            self.state[ self.board_size-1-rand_key[i]//7 ][ self.board_size-1-rand_key[i]%7 ][self.block]=1

        return self._obs((0,))
    
    def step(self, action_dict):# action_dict是双方在该回合的动作
        # 输出为observation, reward, dones, info
        # observation格式为{players[0]:{'obs': ,'action_mask': }, players[1]: {'obs':, 'action_mask': }}
        # reward格式为{players[0]: 1, players[1]: -1}
        # dones格式为{players[0]: 0, players[1]: 0, '__all__': 0} ('__all__'表示是否所有人结束了)
        # info格式为{players[0]: {}, players[1]: {} }（给出额外信息，没有额外信息可以不填）
        assert self.state is not None, "must call reset() first!"
        pos = action_dict[self.players[self.cur_player]]
        x, y = pos // 3, pos % 3
        
        win = -2
        dones = {}
        
        assert not self.state[x][y].any(), 'Invalid action!'
        if self.state[x][y].any():
            # Invalid action!
            win = 1 - self.cur_player
        else:
            self.state[x][y][self.cur_player] = 1
            if (self.state[x, :, self.cur_player].all()
                or self.state[:, y, self.cur_player].all()
                or x == y and self.state[:, :, self.cur_player].diagonal().all()
                or x + y == 2 and self.state[:, ::-1, self.cur_player].diagonal().all()
                ):
                # Win!
                win = self.cur_player
            elif self.state.sum() == 9:
                # Tie!
                win = -1

        if win == -2:
            # continue
            self.cur_player = 1 - self.cur_player
            dones['__all__'] = dones[self.players[self.cur_player]] = False
            return self._obs((self.cur_player,)), {self.players[self.cur_player]: 0}, dones, {self.players[self.cur_player]: {}}
        else:
            # finish
            rewards = {}
            if win == -1:
                for player in self.players: rewards[player] = 0.0
            else:
                rewards[self.players[win]] = 1.0
                rewards[self.players[1 - win]] = -1.0
            dones['__all__'] = True
            for player in self.players: dones[player] = True
            tmp = (self._obs((0, 1)), rewards, dones, {player: {} for player in self.players})
            self.state = None
            return tmp
    
    def _obs(self, players):
        return {self.players[player] : {
            'observation': self.state.copy() if player == 0 else self.state[:,:,::-1].copy(),
            'action_mask': 1 - self.state.sum(-1, dtype = np.int8).flatten()
        } for player in players}

    def render(self): # 可视化
        pass

from ray.tune.registry import register_env
register_env("TicTacToe", TicTacToeEnv)