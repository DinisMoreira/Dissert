import gym
import highway_env
import numpy as np
from matplotlib import pyplot as plt

from stable_baselines import HER, SAC, DDPG, TD3
from stable_baselines.ddpg import NormalActionNoise
from stable_baselines.deepq.dqn import DQN

env = gym.make("parking-v0")

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

# SAC hyperparams:
model = HER('MlpPolicy', env, TD3, n_sampled_goal=n_sampled_goal,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256, policy_kwargs=dict(layers=[256, 256, 256])) #'''tensorboard_log="./TensorBoardLogs/FinalTraining/Training1",'''
            

#model = HER.load('./TrainedAgents/FinalTraining/TD3/training2_1Column_10s_4e5', env=env, tensorboard_log="./TensorBoardLogs/FinalTraining/Training3")
#./TrainedAgents/TD3/BiggerLookAhead/TanVel/1Column_tangentVelocityV3_V2_2e5



model.learn(int(1e5))
model.save('./dummy')

# Load saved model
model = HER.load('./dummy', env=env)

obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(3000):
  action, _ = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
  episode_reward += reward
  if done or info.get('is_success', False):
    print("Reward:", episode_reward, "Success?", info.get('is_success', False))
    episode_reward = 0.0
    obs = env.reset()

print("./TrainedAgents/FinalTraining/TD3/training3_6Column_20s_15e5")
plt.imshow(env.render(mode="rgb_array"))
plt.show()