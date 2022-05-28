import numpy as np
from IPython.display import display

directions=['up','down','left','right']
def watch_state(state_):
    print(f'\nState or observation:{state_.shape}')
    for i in range(3):
        if i == 0 or i == 1:
            print(f'Player_{i}:')
        else:
            print('Wall:')
        print(state_[:,:,i],end='\n')

def watch_mask(mask_):
    print(f'\nAction_mask:{mask_.shape}')
    for i in range(4):
        print(f'{directions[i]}:')
        print(mask_[:,:,i])

def watch_obs(obs_):
    for key in list(obs_.keys()):
        print(f'\n{key}:')
        watch_state(obs_[key]['observation'])
        watch_mask(obs_[key]['action_mask'])

def act_num2tensor(n):
    act=np.zeros(324,dtype=np.int8)
    act[n]=1
    act_=act.reshape(9,9,4)
    watch_mask(act_)
    return act_

def act_tensor2num(m):
    m=m.reshape(324)
    ret=np.where(m)
    display(ret)
    assert len(ret)==1,'action中有多个1'
    assert len(ret[0]==1),'action中有多个1'
    return np.where(m)[0][0]
