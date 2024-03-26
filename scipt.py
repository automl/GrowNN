# %%
import minihack
import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

# %%
env = gym.make(
   "MiniHack-River-v0",
   observation_keys=("pixel", "glyphs", "colors", "chars"),
   max_episode_steps=100,
) 
vec_env = make_vec_env(lambda: env, n_envs=1)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1)


network = model.policy


