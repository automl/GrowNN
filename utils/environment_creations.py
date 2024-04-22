import gym
from typing import List
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import minihack
from nle import nethack


def make_env(env_id: str, observation_keys: List[str], max_episode_steps: int, environment_seed: int):
    """
    From https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#multiprocessing-unleashing-the-power-of-vectorized-environments
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    minihack
    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
        nethack.Command.OPEN,
        nethack.Command.KICK,
        nethack.Command.SEARCH,
    )

    def _init():
        env = gym.make(env_id, observation_keys=observation_keys, max_episode_steps=max_episode_steps, actions=MOVE_ACTIONS)
        # env.reset(seed=environment_seed)
        env.seed(environment_seed, environment_seed)
        return env

    return _init


def make_vec_env(env_id: str, observation_keys: List[str], environment_seed: int, parralel_vec_envs: int, max_episode_steps: int):
    return DummyVecEnv([make_env(env_id, observation_keys, max_episode_steps, environment_seed + i) for i in range(parralel_vec_envs)])
