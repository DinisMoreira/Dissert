import gym
import highway_env
import numpy as np
from matplotlib import pyplot as plt

from stable_baselines import HER, SAC, DDPG, TD3
from stable_baselines.ddpg import NormalActionNoise

from rl_agents.agents.common.factory import agent_factory
from rl_agents.agents.tree_search.mcts import MCTS

import sys

sys.path.insert(0, './highway-env/scripts/')
from utils import record_videos, show_videos, capture_intermediate_frames


# Make environment
env = gym.make("highway-v0")
env = record_videos(env)
obs, done = env.reset(), False
capture_intermediate_frames(env)


# Make agent
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

print(agent.config)

# Run episode
for step in range(env.unwrapped.config["duration"]):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    
env.close()