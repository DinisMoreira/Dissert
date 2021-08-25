import gym
import highway_env
from matplotlib import pyplot as plt
import pprint
'exec(%matplotlib inline)'

env = gym.make('parking-v0')
env.reset()
pprint.pprint(env.config)
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()

plt.imshow(env.render(mode="rgb_array"))
plt.show()
