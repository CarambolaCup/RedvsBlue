from env_working_mid import BlueRedEnv
from model.easy_model import MyModel
from model.deeper_model import LSTMModel

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from ray.tune.registry import ENV_CREATOR, _global_registry

register_env("BlueRed", lambda config: BlueRedEnv(config))
ModelCatalog.register_custom_model("MyModel",MyModel)

from policy import RandomPolicy,FirstValidPolicy
#Trainer
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from gym.spaces import Box,Discrete,Dict,Space
import numpy as np

ray.init()
config=ppo.DEFAULT_CONFIG.copy()
config['env']='BlueRed'
config['env_config']={
    'mid_reward':0.1,
    'final_time':1000,
    'render':False
}

observation_space = Dict(dict(observation=Box(
    0, 1, (9, 9, 3), np.int8), action_mask=Box(0, 1, (324,), np.int8)))
action_space = Discrete(324)

config["_disable_preprocessor_api"] = True
config['framework']='torch'
config['num_gpus']=0
config['gamma']=0.99
config['num_workers']=1
config['num_envs_per_worker']=1
config['rollout_fragment_length']=1000
config['train_batch_size']=4000
config['sgd_minibatch_size']=512
config['num_sgd_iter']=5


config['multiagent']['policies']={
    'random':PolicySpec(policy_class=RandomPolicy,
    observation_space=observation_space,
    action_space=action_space,
    config={}),
    'first_valid':PolicySpec(policy_class=FirstValidPolicy,
    observation_space=observation_space,
    action_space=action_space,
    config={}),
    'train1':PolicySpec(None,observation_space,action_space,{}),
    'train2':PolicySpec(None,observation_space,action_space,{}),
}

def policy_mapping(agent_id, episode, worker, **kwargs):
    if agent_id=='player_1':
        return 'train1'
    else:
        return 'train2'

config['multiagent']['policy_mapping_fn']=policy_mapping

#config['multiagent']['policy_mapping_fn']=lambda agent_id, episode, worker, **kwargs:'ToM1' if agent_id == 'player_1' else 'nearest_hare'
config['multiagent']['policies_to_train']=['train2']
config['model']['custom_model']='MyModel'
config['model']['custom_model_config']['lstm_state_size']=512

render_env_config=config['env_config'].copy()
render_env_config['render']=True
'''
config["evaluation_interval"]=1
config["evaluation_num_episodes"]=5
config['evaluation_config']={
    'render_env':True,
    'env_config':render_env_config
}
'''
trainer=ppo.PPOTrainer(config)
#trainer.restore('D:/College things/新建文件夹/project/partner selection/RedvsBlue-main/result/PPO_BlueRed_2022-05-30_16-14-44qtqgmfjm/checkpoint_000101/checkpoint-101')
trainer.restore('/home/feng/ray_results/PPO_BlueRed_2022-05-30_16-14-44qtqgmfjm/checkpoint_000101/checkpoint-101')
for i in range(1001):
    result=trainer.train()
    print(pretty_print(result))

    if i%50==0:
        checkpoint=trainer.save()
        print("checkpoint saved at", checkpoint)