from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
import torch
from torch import nn

class MyModel(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.lstm_state_size=model_config['custom_model_config']['lstm_state_size']
        self.input_channels=3
        self.world_height=9
        self.world_width=9
        
        self._preprocess=nn.Sequential(
            nn.Conv2d(self.input_channels,4*self.input_channels,3,1,1),
            nn.ReLU(),
            nn.Conv2d(4*self.input_channels,4*self.input_channels,3,1,1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm=nn.LSTM(4*self.input_channels*self.world_height*self.world_width,
        self.lstm_state_size,
        batch_first=True)

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
        self._action_branch=nn.Linear(self.lstm_state_size,324)
        self._value_branch=nn.Linear(self.lstm_state_size,1)
        self._features = None

    def get_initial_state(self):
        return [self._preprocess[0].weight.new(1,self.lstm_state_size).zero_().squeeze(0),
        self._preprocess[0].weight.new(1,self.lstm_state_size).zero_().squeeze(0)]

    def forward_rnn(self,inputs,state,seq_lens):
        obs_flatten=inputs[:,:,324:].float()
        
        print(obs_flatten.shape)
        obs=obs_flatten.reshape(obs_flatten.shape[0],obs_flatten.shape[1],self.world_height,self.world_width,self.input_channels)
        obs=obs.permute(0,1,4,2,3)
        obs_postprocess_set=[]
        for i in range(obs.shape[1]):
            obs_postprocess_set.append(self._preprocess(obs[:,i,...]))
        obs_postprocessed=torch.stack(obs_postprocess_set,dim=1)
        self._features,[h,c]=self.lstm(obs_postprocessed,[torch.unsqueeze(state[0],0),torch.unsqueeze(state[1],0)])

        action_mask=inputs[:,:,:324].float()
        inf_mask=torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
        action_logits=self._action_branch(self._features)
        return action_logits+inf_mask,[torch.squeeze(h,0),torch.squeeze(c,0)]

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self._value_branch(self._features),[-1])