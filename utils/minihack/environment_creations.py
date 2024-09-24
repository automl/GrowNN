import gym
from typing import List, Optional
from stable_baselines3.common.vec_env import DummyVecEnv
import minihack
from nle import nethack


def make_minihack_env(env_id: str, observation_keys: List[str], max_episode_steps: Optional[int], environment_seed: int):
    """
    Create a MiniHack environment with the given parameters
    :param env_id: The id of the environment
    :type env_id: str
    :param observation_keys: The keys of the observations
    :type observation_keys: List[str]
    :param max_episode_steps: The maximum number of steps per episode
    :type max_episode_steps: Optional[int]
    :param environment_seed: The seed of the environment
    :type environment_seed: int
    """
    minihack
    MOVE_ACTIONS = tuple(nethack.CompassDirection)

    def _init():
        nonlocal max_episode_steps
        if max_episode_steps is None:
            env = gym.make(env_id, observation_keys=observation_keys, actions=MOVE_ACTIONS)
        else:
            env = gym.make(env_id, observation_keys=observation_keys, max_episode_steps=max_episode_steps, actions=MOVE_ACTIONS)
        # Force Create a Random Number Generator in MiniHack. I added this to ensure reproducibility
        env.create_np_rng(environment_seed)
        return env

    return _init


def make_minihack_vec_env(env_id: str, observation_keys: List[str], environment_seed: int, parralel_vec_envs: int, max_episode_steps: int):
    """
    Create a MiniHack vectorized environment with the given parameters
    :param env_id: The id of the environment
    :type env_id: str
    :param observation_keys: The keys of the observations
    :type observation_keys: List[str]
    :param environment_seed: The seed of the environment
    :type environment_seed: int
    :param parralel_vec_envs: The number of parallel environments
    :type parralel_vec_envs: int
    :param max_episode_steps: The maximum number of steps per episode
    :type max_episode_steps: int
    """
    return DummyVecEnv([make_minihack_env(env_id, observation_keys, max_episode_steps, environment_seed + i) for i in range(parralel_vec_envs)])
