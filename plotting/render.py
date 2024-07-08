# %%
import gym
from matplotlib import pyplot as plt
import minihack
# %%
env = gym.make("MiniHack-Room-Ultimate-15x15-v0", observation_keys=("pixel",))
obs = env.reset()

# %%

# %%
obs["pixel"].shape

# %%
plt.imshow(obs["pixel"])
plt.show()
# %%
