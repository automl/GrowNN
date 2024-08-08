import gym
from typing import List, Optional
from stable_baselines3.common.vec_env import DummyVecEnv
import minihack
from nle import nethack
from stable_baselines3.common.env_util import make_vec_env

def make_minihack_env(env_id: str, observation_keys: List[str], max_episode_steps: Optional[int], environment_seed: int):
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

    def _init():
        nonlocal max_episode_steps
        if max_episode_steps is None:
            # if "Room" in env_id:
            #    grid_size = int(env_id.split("-")[-2].split("x")[0])
            #    max_episode_steps = grid_size * grid_size * 4
            #    env = gym.make(env_id, observation_keys=observation_keys, max_episode_steps = max_episode_steps,actions=MOVE_ACTIONS)
            # else:
            env = gym.make(env_id, observation_keys=observation_keys, actions=MOVE_ACTIONS)
        else:
            env = gym.make(env_id, observation_keys=observation_keys, max_episode_steps=max_episode_steps, actions=MOVE_ACTIONS)
        # env.reset(seed=environment_seed)
        env.create_np_rng(environment_seed)
        return env

    return _init


def make_minihack_vec_env(env_id: str, observation_keys: List[str], environment_seed: int, parralel_vec_envs: int, max_episode_steps: int):
    return DummyVecEnv([make_minihack_env(env_id, observation_keys, max_episode_steps, environment_seed + i) for i in range(parralel_vec_envs)])

def make_bipedal_walker_vec_env(env_id: str, environment_seed: int, parralel_vec_envs: int):
    return make_vec_env(env_id, n_envs=parralel_vec_envs, seed=environment_seed, vec_env_cls=DummyVecEnv)