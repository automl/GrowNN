import os
from functools import partial
import gym
import hydra
import minihack
import numpy as np
import torch
from ConfigSpace import Configuration
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO

from py_experimenter.result_processor import ResultProcessor
from utils import create_pyexperimenter, extract_hyperparameters, log_results, make_vec_env, get_model_save_path, config_is_evaluated, get_budget_path_dict
from utils.networks.feature_extractor import Net2DeeperFeatureExtractor
from utils.stable_baselines_callback import CustomEvaluationCallback, FinalEvaluationWrapper

debug_mode = False


@hydra.main(config_path="config", config_name="net2wider", version_base="1.1")
def net2wider_configure(config: Configuration):
    def net2wider_execute(result_processor: ResultProcessor):
        # Mention the used libraries because of implicit imports
        minihack
        gym

        seed = config["seed"]
        set_random_seed(seed, using_cuda=True)
        # Idea save to n_trials_seed_budget_hpohash
        # To find the current seed, we ignore n_trials but select based on the rest
        # Question: How do we log? We write the id into the log file. But how do we know which run is continued where

        # TODO Load previously trained model
        non_hyperparameters = config["non_hyperparameters"]
        environment_name = non_hyperparameters["environment_name"]
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
            feature_extractor_output_dimension,
            n_feature_extractor_layers,
            feature_extractor_layer_width,
        ) = extract_hyperparameters(config)

        feature_extractor_layer_width = int(feature_extractor_layer_width)

        hyperparameter_str_identifier = str(extract_hyperparameters(config))

        # Todo rebuild the convert space functionality from stablebaselines to work with a reliable gym env
        # https://github.com/DLR-RM/stable-baselines3/blob/5623d98f9d6bcfd2ab450e850c3f7b090aef5642/stable_baselines3/common/vec_env/patch_gym.py#L63

        # We always use the same seeds in here
        training_vec_env = make_vec_env(
            environment_name,
            non_hyperparameters["observation_keys"],
            non_hyperparameters["env_seed"],
            non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["max_episode_steps"],
        )

        # Check whether to wrap in monitor wrapper
        evaluation_vec_env = make_vec_env(
            environment_name,
            non_hyperparameters["observation_keys"],
            non_hyperparameters["env_seed"] + non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["max_episode_steps"],
        )
        torch.cuda.torch.cuda.empty_cache()

        feature_extractor = partial(
            Net2DeeperFeatureExtractor,
            cnn_intermediate_dimension=1,
            n_feature_extractor_layers=n_feature_extractor_layers,
            feature_extractor_layer_width=feature_extractor_layer_width,
            feature_extractor_output_dimension=feature_extractor_output_dimension,
        )

        model = PPO(
            policy="MultiInputPolicy",
            env=training_vec_env,
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
            policy_kwargs={"features_extractor_class": feature_extractor, "net_arch": {"pi": [feature_extractor_output_dimension], "vf": [feature_extractor_output_dimension]}},
        )
        # TODO Adapt the budget to not be feature_extractor_depth but feature_extractor_width

        if config_is_evaluated(config.non_hyperparameters.model_save_path, config):
            x = get_budget_path_dict(config.non_hyperparameters.model_save_path, config)

        evaluation_callback = CustomEvaluationCallback(
            evaluation_vec_env,
            n_eval_episodes=non_hyperparameters["n_evaluation_episodes"],
            eval_freq=non_hyperparameters["total_timesteps"] / non_hyperparameters["n_evaluation_rounds"] / non_hyperparameters["parallel_vec_envs"],
            deterministic=True,
            render=False,
            log_path="./logs",
        )
        model.learn(total_timesteps=non_hyperparameters["total_timesteps"], callback=evaluation_callback)
        if not debug_mode:
            evaluation_callback.log_results(result_processor, non_hyperparameters["trial_number"], seed, ent_coef, vf_coef, hyperparameter_str_identifier=hyperparameter_str_identifier)

        # TODO Save the model and feature extractor
        final_save_path = get_model_save_path(config.non_hyperparameters.model_save_path, config, feature_extractor_layer_width, seed)
        if not os.path.exists(final_save_path):
            os.makedirs(final_save_path)

        model.save(os.path.join(final_save_path, "model"))

        if not debug_mode:
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
                            "budget": feature_extractor_layer_width,
                            "hyperparameter_str_identifier": hyperparameter_str_identifier,
                            "timestep": timestep,
                            "evaluated_cost": evaluated_cost,
                            "evaluated_stdev": evalauted_stdev,
                        }
                    },
                )

        evaluation_vec_env = make_vec_env(
            environment_name,
            non_hyperparameters["observation_keys"],
            non_hyperparameters["env_seed"] + non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["max_episode_steps"],
        )

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
        if not debug_mode:
            callback_wrapper.process_results(
                non_hyperparameters["trial_number"], seed, final_score, final_std, budget=feature_extractor_layer_width, hyperparameter_str_identifier=hyperparameter_str_identifier
            )

            log_results(
                result_processor,
                {
                    "configurations": {
                        "worker_number": seed,  # Currently the same as the workerseed
                        "worker_seed": seed,
                        "trial_number": non_hyperparameters["trial_number"],
                        "budget": feature_extractor_layer_width,
                        "hyperparameter_str_identifier": hyperparameter_str_identifier,
                        "environment_name": environment_name,
                        "batch_size": batch_size,
                        "clip_range": clip_range,
                        "clip_range_vf": clip_range_vf,
                        "ent_coef": ent_coef,
                        "gae_lambda": gae_lambda,
                        "learning_rate": learning_rate,
                        "max_grad_norm": max_grad_norm,
                        "n_epochs": n_epochs,
                        "n_steps": n_steps,
                        "normalize_advantage": normalize_advantage,
                        "vf_coef": vf_coef,
                        "final_score": final_score,
                        "final_std": final_std,
                    }
                },
            )

        model.policy = None
        torch.cuda.empty_cache()
        return float(-final_score)

    if debug_mode:
        return net2wider_execute(None)
    else:
        # Attach Execution to pyExperimetner to then enable logging
        experimenter = create_pyexperimenter(config)
        result = experimenter.attach(net2wider_execute, config.non_hyperparameters.experiment_id)
        return float(result)


if __name__ == "__main__":
    net2wider_configure()
