import gym
import minihack
from ConfigSpace import Configuration, ConfigurationSpace, Float
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
import hydra
import logging


@hydra.main(config_path="configs", config_name="blackbox_ppo", version_base="1.1")
def black_box_ppo(config: Configuration, seed: int = 0):
    # TODO use config
    # TODO use 
    with open('something_executed.txt', 'w') as file:
        # Write some content into the file
        file.write('Hello, world!\n')
        file.write('This is a test file.\n')
        file.write('Writing into files using Python is easy!\n')
    env = gym.make(
        "MiniHack-River-v0",
        observation_keys=("pixel", "glyphs", "colors", "chars"),
        max_episode_steps=100,
    ) 
    vec_env = make_vec_env(lambda: env, n_envs=1)
    model = PPO("MultiInputPolicy", vec_env, verbose=1, device="cuda")
    model.learn(total_timesteps=1000)
    total_rewards = []
    for _ in range(5):
        obs = vec_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    average_reward = sum(total_rewards) / len(total_rewards)
    return - average_reward

if __name__ == "__main__":
    black_box_ppo()