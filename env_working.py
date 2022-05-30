from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Box, Discrete, Dict
import numpy as np
import random
import gif
import matplotlib.pyplot as plt
import time
import os

class BlueRedEnv(MultiAgentEnv):
    observation_space = Dict(dict(observation=Box(
        0, 1, (9, 9, 3), np.int8), action_mask=Box(0, 1, (324,), np.int8)))
    action_space = Discrete(324)
    players = ('player_1', 'player_2')
    directions = ['up', 'down', 'left', 'right']

    def __init__(self, config):
        self.state = None
        self.board_size = 8
        self.width=9
        self.height=9
        # self.players_1 = 0
        # self.players_2 = 1
        self.block = 2
        self.turns = None

        self.mid_reward=config['mid_reward']
        self.final_time=config['final_time']

        
        self.render_env=config['render']
        self.render_count=0
        self.render_frame=[]
        self.dir_name=int(time.time())

    def reset(self):
        self.turns = 0
        self.state = np.zeros((9, 9, 3), dtype=np.int8)
        self.cur_player = 0
        player_init_place = np.array([[0, 0], [0, 4], [0, 8], [5, 8], [
                                     5, 0], [8, 2], [8, 6]])  # 棋子初始位置
        for i in range(0, 7):  # 把两种颜色的棋子放到棋盘上
            self.state[player_init_place[i][0]][player_init_place[i][1]][0] = 1
            self.state[self.board_size - player_init_place[i][0]
                       ][self.board_size - player_init_place[i][1]][1] = 1
        rand_key = random.sample(range(0, 24), 2)
        for i in range(0, 2):
            self.state[1+rand_key[i] // 7][1+rand_key[i] % 7][self.block] = 1
            self.state[self.board_size-1-rand_key[i] //
                       7][self.board_size-1-rand_key[i] % 7][self.block] = 1

        if self.render_env:
            if not os.path.exists(f'./gif/exp{self.dir_name}'):
                os.makedirs(f'./gif/exp{self.dir_name}')
            if len(self.render_frame)!=0:
                self.render_count+=1
                gif.save(self.render_frame,path=f'./gif/exp{self.dir_name}/gif{self.render_count}.gif',duration=100,unit='milliseconds',between='frames',loop=True)
                self.render_frame=[]
            self.render()

        return self._obs((0,))

    def step(self, action_dict):  # action_dict是双方在该回合的动作
        # 输出为observation, reward, dones, info
        # observation格式为{players[0]:{'obs': ,'action_mask': }, players[1]: {'obs':, 'action_mask': }}
        # reward格式为{players[0]: 1, players[1]: -1}
        # dones格式为{players[0]: 0, players[1]: 0, '__all__': 0} ('__all__'表示是否所有人结束了)
        # info格式为{players[0]: {}, players[1]: {} }（给出额外信息，没有额外信息可以不填）
        # assert self.state is not None, "must call reset() first!"
        # action_dict即为由AI搞出来的当前这一步动作，一个玩家为一步，并不是两个玩家各走一步(lyj)
        # 以500回合（双方各500个动作）为界限来判定是否为平局，如果超过500回合则我们判定为平局，其余必有输赢(lyj)
        pos = action_dict[self.players[self.cur_player]]  # 玩家刚才走过的动作，以一个值表示

        # 将动作的值改为9*9*4的动作矩阵（lyj）
        act = self.act_num2tensor(pos)  # （lyj）
        line, lie, direction = 0, 0, 0  # (lyj)
        # (lyj)搞出动的行、列、方向
        for i in range(9):
            for j in range(9):
                for k in range(4):
                    if(act[i][j][k] != 0):
                        line, lie, direction = i, j, k
        win = -2
        dones = {}
        self.turns += 1  # 回合数增加1（lyj）

        # assert not self.state[line][lie][1-self.cur_player]!=0, 'Invalid action!'
        # （lyj）
        # ?????
        rewards={'player_1':0,'player_2':0}
        self.state[line][lie][self.cur_player] = 0
        if direction == 0:
            self.state[line-1][lie][self.cur_player] = 1
        elif direction == 1:
            self.state[line+1][lie][self.cur_player] = 1
        elif direction == 2:
            self.state[line][lie-1][self.cur_player] = 1
        elif direction == 3:
            self.state[line][lie+1][self.cur_player] = 1

        for i in range(9):
            for j in range(9):
                count = 0
                if i >= 1 and self.state[i-1][j][self.cur_player] == 1:
                    count += 1
                if j >= 1 and self.state[i][j-1][self.cur_player] == 1:
                    count += 1
                if j <= 7 and self.state[i][j+1][self.cur_player] == 1:
                    count += 1
                if i <= 7 and self.state[i+1][j][self.cur_player] == 1:
                    count += 1
                if count >= 2 and self.state[i][j][1-self.cur_player]==1:
                    self.state[i][j][1-self.cur_player] = 0
                    rewards[self.players[self.cur_player]]+=self.mid_reward
                

        for i in range(9):
            for j in range(9):
                count = 0
                if i >= 1 and self.state[i-1][j][1-self.cur_player] == 1:
                    count += 1
                if j >= 1 and self.state[i][j-1][1-self.cur_player] == 1:
                    count += 1
                if j <= 7 and self.state[i][j+1][1-self.cur_player] == 1:
                    count += 1
                if i <= 7 and self.state[i+1][j][1-self.cur_player] == 1:
                    count += 1
                if count >= 2 and self.state[i][j][self.cur_player]==1:
                    self.state[i][j][self.cur_player] = 0
                    rewards[self.players[1-self.cur_player]]+=self.mid_reward


        finished = True
        for i in range(9):
            for j in range(9):
                if(self.state[i][j][1-self.cur_player] != 0):
                    finished = False
        if (finished):
            # Win!
            win = self.cur_player
        elif self.turns == self.final_time:  # 两个人一共走了1000回合（lyj）
            # Tie!
            win = -1

        if win == -2:
            # continue,not finished
            self.cur_player = 1 - self.cur_player
            # dones(lyj)
            dones['__all__'] = dones[self.players[self.cur_player]] = False
            return self._obs((self.cur_player,)), rewards, dones, {self.players[self.cur_player]: {}}
        else:
            # finish,has a result
            # write rewards
            # rewards = {}
            if win == -1:
                for player in self.players:
                    rewards[player] += 0
            else:
                rewards[self.players[win]] += 1.0
                rewards[self.players[1 - win]] += -1.0
            dones['__all__'] = True
            for player in self.players:
                dones[player] = True
            tmp = (self._obs((0, 1)), rewards, dones, {
                   player: {} for player in self.players})
            return tmp

    def _obs(self, players_):
        # 返回一个dict，具体看kaggle_test_notebook
        # observation: 9*9*3 第一个9*9是己方，第二个是对方，第三个是障碍物，称为‘Wall’
        # action_mask: 9*9*4 代表上下左右
        d0 = np.array([-1, 1, 0, 0], dtype=np.int8)
        d1 = np.array([0, 0, -1, 1], dtype=np.int8)

        def out_of_range(x, y):
            if x < 0 or x > 8 or y < 0 or y > 8:
                return True
            return False
        ret = dict()
        wall_and_pieces = self.state.sum(-1, dtype=np.int8)

        for player in players_:
            ob = self.state.copy() if player == 0 else self.state[:, :, [
                1, 0, 2]].copy()

            mask_list = []  # 4个9*9矩阵分别对应上下左右
            for i in range(4):
                m_init = ob[:, :, 0].copy()
#                 display(m_init)
                for x0 in range(9):
                    for x1 in range(9):
                        if m_init[x0][x1] == 0:
                            continue
                        if out_of_range(x0+d0[i], x1+d1[i]) or wall_and_pieces[x0+d0[i]][x1+d1[i]]:
                            m_init[x0][x1] = 0
                mask_list.append(m_init)
#                 display(m_init)
            action_mask = np.stack(mask_list, axis=0).transpose(1, 2, 0)
            ret[self.players[player]] = {
                'observation': ob,
                'action_mask': action_mask.reshape(324)}
        return ret

    def act_num2tensor(self,n):
        act=np.zeros(324,dtype=np.int8)
        act[n]=1
        act_=act.reshape(9,9,4)
        return act_

    def render(self):
        @gif.frame
        def plot():
            vlines=np.linspace(-0.5,-0.5+self.width,self.width+1)
            hlines=np.linspace(-0.5,-0.5+self.height,self.height+1)
            plt.hlines(hlines,-0.5,-0.5+self.width)
            plt.vlines(vlines,-0.5,-0.5+self.height)
            plt.axis('off')
            block_pos=np.nonzero(self.state[:,:,2])
            blue_pos=np.nonzero(self.state[:,:,1])
            red_pos=np.nonzero(self.state[:,:,0])
        
            for i in range(len(block_pos[0])):
                y=block_pos[0][i]
                x=block_pos[1][i]
                plt.fill([x+0.5,x+0.5,x-0.5,x-0.5],[y-0.5,y+0.5,y+0.5,y-0.5],'black')
            for i in range(len(blue_pos[0])):
                y=blue_pos[0][i]
                x=blue_pos[1][i]
                plt.fill([x+0.5,x+0.5,x-0.5,x-0.5],[y-0.5,y+0.5,y+0.5,y-0.5],'blue')
            for i in range(len(red_pos[0])):
                y=red_pos[0][i]
                x=red_pos[1][i]
                plt.fill([x+0.5,x+0.5,x-0.5,x-0.5],[y-0.5,y+0.5,y+0.5,y-0.5],'red')
        self.render_frame.append(plot())
        plt.close()
        return True


