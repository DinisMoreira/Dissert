import gym
import highway_env
import numpy as np
from matplotlib import pyplot as plt
import pprint

from stable_baselines import HER, SAC, DDPG, TD3
from stable_baselines.ddpg import NormalActionNoise

import time

env = gym.make("parking-v0")
pprint.pprint(env.config)
success_accumulator = 0
episodes_accumulator = 0

# Load saved model
model = HER.load('./TrainedAgents/FinalTraining/TD3/training3_6Column_20s_15e5', env=env)

obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(50000):
  action, _ = model.predict(obs)
  #startTime = time.time()
  obs, reward, done, info = env.step(action)
  #endTime = time.time()
  #print(endTime-startTime)
  env.render()
  episode_reward += reward
  if done or info.get('is_success', False):
    print("Reward:", episode_reward, "Success?", info.get('is_success', False))

    if info.get('is_success', False):
      success_accumulator += 1
    episodes_accumulator += 1

    episode_reward = 0.0
    obs = env.reset()

print("./TrainedAgents/FinalTraining/TD3/training3_6Column_20s_15e5")
print("SUCCESS RATE = " + str(success_accumulator/episodes_accumulator))
plt.imshow(env.render(mode="rgb_array"))
plt.show()