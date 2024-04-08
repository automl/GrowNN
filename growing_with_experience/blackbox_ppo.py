from ConfigSpace import Configuration
import gym
import gymnasium
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

def black_box_ppo(config: Configuration, seed: int = 0):
    num_cpu = 4
    env_id = "CartPole-v1"
    env = gym.make(
        "MiniHack-River-v0",
        observation_keys=("pixel", "glyphs", "colors", "chars"),
        max_episode_steps=100,
    ) 
    vec_env = make_vec_env(lambda: env, n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=1)
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
    return average_reward