# 这是一个写policy的示例，只用修改MyPolicy里的compute_actions
# 就是给你一个batch的obs（比如说是一个包含32个obs的列表）
# 然后你返回给它一个batch的action
# 实际上就是你要对每个obs给出一个action。
# 只是系统会一次性给你batch_size个obs，然后你也一次性给它返回batch_size个action
from ray.rllib.policy.policy import Policy
import numpy as np


class MyPolicy(Policy):

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
