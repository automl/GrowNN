from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env


def make_bipedal_walker_vec_env(env_id: str, environment_seed: int, parralel_vec_envs: int, hardcore: bool):
    return make_vec_env(env_id, n_envs=parralel_vec_envs, seed=environment_seed, vec_env_cls=DummyVecEnv, env_kwargs={"hardcore": hardcore})

def make_ant_vec_env(env_id: str, environment_seed: int, parralel_vec_envs: int, hardcore: bool):
    return make_vec_env(env_id, n_envs=parralel_vec_envs, seed=environment_seed, vec_env_cls=DummyVecEnv)