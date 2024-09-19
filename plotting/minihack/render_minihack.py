# %%
import gym
from matplotlib import pyplot as plt
import minihack
import numpy as np
# %%
env = gym.make("MiniHack-Room-Monster-5x5-v0", observation_keys=("pixel","chars",))
obs = env.reset()

# %%

# %%
obs["pixel"].shape

# %%
plt.imshow(obs["pixel"])
plt.show()
# %%
plt.imshow(obs["chars"])
np.set_printoptions(threshold=np.inf)
print(obs["chars"])
# %%
