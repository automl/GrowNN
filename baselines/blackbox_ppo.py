import gym
import minihack
from ConfigSpace import Configuration, Float
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.ppo import PPO
import hydra
from utils import make_vec_env, make_env, extract_hyperparameters, create_pyexperimenter, log_results
from py_experimenter.result_processor import ResultProcessor
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from utils.stable_baseliens_callback import FinalEvaluationWrapper

# Next step is the callback during training
# Also keep an eye on the data that we generate in the currently running job
# Mit Theresa darüber reden ob es eine art liit an der menge an jobs gibt ,die submitted werden können. Denn gestern hat 3113145_0 nicht gestartet
# Obwohl ressourcen verfügbar schienen
# TODO plots über den learning process erstellen


@hydra.main(config_path="config", config_name="blackbox_ppo", version_base="1.1")
def black_box_ppo_configure(config: Configuration):
    def black_box_ppo_execute(result_processor: ResultProcessor):
        minihack
        gym
        # TODO are the inptu features noramlized?

        seed = config["seed"]
        # TODO Build Reward Manager https://minihack.readthedocs.io/en/latest/getting-started/reward.html
        # TODO only use navigation actions
        # TODO How do we continue training from a previous point, or do we simply restart training with a higher fidelity? To be determined in next version
        non_hyperparameters = config["non_hyperparameters"]
        (
            batch_size,
            clip_range,
            clip_range_vf,
            ent_coef,
            gae_lambda,
            learning_rate,
            max_grad_norm,
            n_epochs,
            n_steps,
            normalize_advantage,
            vf_coef,
        ) = extract_hyperparameters(config)

        # Todo rebuild the convert space functionality from stablebaselines to work with a reliable gym env
        # https://github.com/DLR-RM/stable-baselines3/blob/5623d98f9d6bcfd2ab450e850c3f7b090aef5642/stable_baselines3/common/vec_env/patch_gym.py#L63

        # We only seed the neural network. Everything else is seeded more or less constantly
        set_random_seed(seed, using_cuda=True)

        # We always use the same seeds in here
        training_vec_env = make_vec_env(
            non_hyperparameters["environment_id"],
            non_hyperparameters["observation_keys"],
            non_hyperparameters["env_seed"],
            non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["max_episode_steps"],
        )

        # Check whether to wrap in monitor wrapper
        evaluation_vec_env = make_vec_env(
            non_hyperparameters["environment_id"],
            non_hyperparameters["observation_keys"],
            non_hyperparameters["env_seed"] + non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["max_episode_steps"],
        )

        model = PPO(
            "MultiInputPolicy",
            training_vec_env,
            verbose=2,
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
            n_steps=n_steps,  # The number of steps to run for each environment per update
            seed=seed,
        )

        # TODO set eval_freq on parameters
        evaluation_callback = EvalCallback(
            evaluation_vec_env,
            n_eval_episodes=non_hyperparameters["n_evaluation_episodes"],
            eval_freq=non_hyperparameters["total_timesteps"] / non_hyperparameters["n_evaluation_rounds"],
            deterministic=True,
            render=False,
            log_path="./logs",
        )
        new_logger = configure("logs", ["stdout", "csv"])

        model.set_logger(new_logger)

        model.learn(total_timesteps=non_hyperparameters["total_timesteps"], callback=evaluation_callback)
        callback_data = np.load("logs/evaluations.npz")

        for timestep, result, _ in zip(*callback_data.values()):
            # Check whether we evalaute 10 episodes
            evaluated_cost = np.mean(result)
            evalauted_stdev = np.std(result)
            log_results(
                result_processor,
                {
                    "training_process": {
                        "worker_id": seed,
                        "trial_number": non_hyperparameters["trial_number"],
                        "timestep": timestep,
                        "evaluated_cost": evaluated_cost,
                        "evaluated_stdev": evalauted_stdev,
                    }
                },
            )

        # TODO track solutionrate during learning
        # TODO Evaluation Videos
        # TODO Track solutionrate
        # TODO track loss components
        # TODO check reward conversion

        evaluation_vec_env = make_vec_env(
            non_hyperparameters["environment_id"],
            non_hyperparameters["observation_keys"],
            non_hyperparameters["env_seed"] + non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["max_episode_steps"],
        )

        # TODO use a not vectorized environemnt here
        callback_wrapper = FinalEvaluationWrapper(
            result_processor,
            non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["n_evaluation_episodes"],
        )
        final_score, final_std = evaluate_policy(
            model,
            evaluation_vec_env,
            n_eval_episodes=non_hyperparameters["n_evaluation_episodes"],
            deterministic=True,
            render=False,
            callback=callback_wrapper.get_callback(),
        )

        callback_wrapper.process_results(non_hyperparameters["trial_number"], seed, final_score, final_std)

        log_results(
            result_processor,
            {
                "configurations": {
                    "worker_number": seed,  # Currently the same as the workerseed
                    "worker_seed": seed,
                    "trial_number": non_hyperparameters["trial_number"],
                    "environment_id": non_hyperparameters["environment_id"],
                    "batch_size": batch_size,
                    "clip_range": clip_range,
                    "clip_range_vf": clip_range_vf,
                    "ent_coef": ent_coef,
                    "gae_lambda": gae_lambda,
                    "learning_rate": learning_rate,
                    "max_grad_norm": max_grad_norm,
                    "n_epochs": n_epochs,
                    "normalize_advantage": normalize_advantage,
                    "vf_coef": vf_coef,
                    "final_score": final_score,
                    "final_std": final_std,
                }
            },
        )
        return float(-final_score)

    # Attach Execution to pyExperimetner to then enable logging
    experimenter = create_pyexperimenter(config, use_ssh_tunnel=True)
    result = experimenter.attach(black_box_ppo_execute, config.non_hyperparameters.experiment_id)
    return float(result)
    # TODO check calback


if __name__ == "__main__":
    black_box_ppo_configure()
