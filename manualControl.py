import gym
import highway_env
import numpy as np
from matplotlib import pyplot as plt
import pprint

env = gym.make("parking-v0")
env.configure({
    "manual_control": True
})
env.reset()
done = False
while not done:
    env.step(env.action_space.sample())  # with manual control, these actions are ignored
    env.render()
    