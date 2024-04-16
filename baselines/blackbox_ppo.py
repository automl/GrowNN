import gym
import minihack
from ConfigSpace import Configuration, Float
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO
import hydra
from utils import make_vec_env, make_env, extract_hyperparameters
from py_experimenter.experimenter import PyExperimenter



@hydra.main(config_path="config", config_name="blackbox_ppo", version_base="1.1")
def black_box_ppo(config: Configuration, seed: int = 0):
    minihack
    gym

    # TODO Build Reward Manager https://minihack.readthedocs.io/en/latest/getting-started/reward.html

    # TODO How do we continue training from a previous point, or do we simply restart training with a higher fidelity? To be determined in next version
    non_hyperparameters = config["non_hyerparameters"]
    (batch_size, clip_range, clip_range_vf, ent_coef, gae_lambda, learning_rate, max_grad_norm, n_epochs, normalize_advantage, vf_coef) = (
        extract_hyperparameters(config)
    )

    # Todo rebuild the convert space functionality from stablebaselines to work with a reliable gym env
    # https://github.com/DLR-RM/stable-baselines3/blob/5623d98f9d6bcfd2ab450e850c3f7b090aef5642/stable_baselines3/common/vec_env/patch_gym.py#L63
    set_random_seed(seed, using_cuda=True)
    vec_env = make_vec_env(
        non_hyperparameters["env_id"],
        non_hyperparameters["observation_keys"],
        seed,
        non_hyperparameters["parralel_vec_envs"],
        non_hyperparameters["max_episode_steps"],
    )
    # Current Mission: Find out which steps does what
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        device="cuda",
        batch_size=batch_size,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        n_epochs=n_epochs,
        normalize_advantage=normalize_advantage,
        vf_coef=vf_coef,
        n_steps=non_hyperparameters["n_steps"],  # The number of steps to run for each environment per update
        seed=seed,
    )
    model.learn(total_timesteps=10000)
    total_rewards = []

    for evaluation_seed in range(10):
        evaluation_env = make_env(
            non_hyperparameters["env_id"],
            non_hyperparameters["observation_keys"],
            non_hyperparameters["max_episode_steps"],
            environment_seed=evaluation_seed,
        )()
        evaluation_env.seed(evaluation_seed, evaluation_seed)
        obs = evaluation_env.reset()

        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = evaluation_env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    average_reward = sum(total_rewards) / len(total_rewards)
    return -average_reward


if __name__ == "__main__":
    experimenter = PyExperimenter("baselines/config/blackbox_ppo.yaml", use_ssh_tunnel=True, use_codecarbon=False)

    # Currenlty I always delete the table in the beginning, because I am in development and will therefore chagne the table a lot
    experimenter.delete_table()
    experimenter.create_table()

    # Execute the Main Code
    black_box_ppo()
