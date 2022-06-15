from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimConv2d,SlimFC
from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
import torch
from torch import nn

class DenseModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self,obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self._conv=nn.Sequential(
            nn.Conv2d(3,8,3,1,1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*9*9,1024),
            nn.ReLU()
        )
        self._logits=nn.Sequential(
            nn.Linear(1024,324)
        )
        self._value_branch=nn.Sequential(
            nn.Linear(1024,1)
        )
        self._hidden = None

    def forward(self, input_dict,state, seq_lens):
        obs=input_dict['obs']['observation'].float()
        obs=obs.permute(0,3,1,2)
        self._hidden=self._conv(obs)
        action_logits=self._logits(self._hidden)
        action_mask=input_dict['obs']['action_mask'].float()
        inf_mask=torch.clamp(torch.log(action_mask),FLOAT_MIN,FLOAT_MAX)
        return action_logits+inf_mask, state

    def value_function(self):
        assert self._hidden is not None, "must call forward() first"
        x=torch.squeeze(self._value_branch(self._hidden),1)
        return torch.squeeze(self._value_branch(self._hidden),1)