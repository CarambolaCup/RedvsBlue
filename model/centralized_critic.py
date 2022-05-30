from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
import torch
from torch import nn
import ray


class MyModel(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.lstm_state_size=model_config['custom_model_config']['lstm_state_size']
        self.input_channels=model_config['custom_model_config']['input_channels']
        self.world_height=model_config['custom_model_config']['world_height']
        self.world_width=model_config['custom_model_config']['world_width']
        self.player_num=model_config['custom_model_config']['player_num']
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            #self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            
        self._preprocess=nn.Sequential(
            nn.Conv2d(self.input_channels,4*self.input_channels,3,1,1),
            nn.ReLU(),
            nn.Conv2d(4*self.input_channels,2*self.input_channels,3,1,1),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)
        self.lstm=nn.LSTM(2*self.input_channels*self.world_height*self.world_width,
        self.lstm_state_size,
        batch_first=True).to(self.device)
        
        '''
        self._preprocess=nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.world_height*self.world_width*self.input_channels,128),
            nn.ReLU()
        )
        self.lstm=nn.LSTM(128,
        self.lstm_state_size,
        batch_first=True)
        '''
        self._action_branch=nn.Linear(self.lstm_state_size,7).to(self.device)
        #self._value_branch=nn.Linear(self.lstm_state_size,1).to(self.device)
        
        self._value_branch=nn.Sequential(nn.Linear(self.lstm_state_size+7*(self.player_num-1),256),
        nn.ReLU(),
        nn.Linear(256,1)).to(self.device)
        
        self._features = None
        self._add_action_features=None

    def get_initial_state(self):
        return [self._preprocess[0].weight.new(1,self.lstm_state_size).zero_().squeeze(0),
        self._preprocess[0].weight.new(1,self.lstm_state_size).zero_().squeeze(0)]

    def forward_rnn(self,inputs,state,seq_lens):
        inputs = inputs.to(self.device)
        env_size=self.input_channels*self.world_height*self.world_width
        state = [state[0].to(self.device), state[1].to(self.device)]
        #obs_flatten=inputs[:,:,7:].float()
        #obs=obs_flatten.reshape(obs_flatten.shape[0],obs_flatten.shape[1],self.world_height,self.world_width,self.input_channels)
        #print(inputs.shape)
        #print('AAA')
        #print(torch.sum(inputs[0,0,:]))
        obs_flatten=inputs[:,:,7:7+env_size].float()
        obs=obs_flatten.reshape(obs_flatten.shape[0],obs_flatten.shape[1],self.world_height,self.world_width,self.input_channels)
        obs=obs.permute(0,1,4,2,3)
        obs_postprocess_set=[]
        for i in range(obs.shape[1]):
            obs_postprocess_set.append(self._preprocess(obs[:,i,...]))
        obs_postprocessed=torch.stack(obs_postprocess_set,dim=1)
        self._features,[h,c]=self.lstm(obs_postprocessed,[torch.unsqueeze(state[0],0),torch.unsqueeze(state[1],0)])

        self._add_action_features=torch.cat((self._features,inputs[:,:,7+env_size:]),2)
        action_mask=inputs[:,:,:7].float()
        inf_mask=torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
        action_logits=self._action_branch(self._features)
        return (action_logits+inf_mask).cpu(), [torch.squeeze(h,0).cpu(),torch.squeeze(c,0).cpu()]

    def value_function(self):
        return torch.reshape(self._value_branch(self._add_action_features),[-1]).cpu()