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