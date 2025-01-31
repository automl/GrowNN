import os
from functools import partial
import gym
import hydra
import minihack
import torch
from ConfigSpace import Configuration
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO

from py_experimenter.result_processor import ResultProcessor
from utils import create_pyexperimenter, extract_hyperparameters_minihack, log_results, make_minihack_vec_env, get_model_save_path_minihack
from utils.minihack.feature_extractor import Net2DeeperFeatureExtractor
from utils.stable_baselines_callback import CustomEvaluationCallback, FinalEvaluationWrapper

debug_mode = False
budget_level_dict: dict = {
    1: 10,
    2: 10,
    3: 10,
    4: 15,
}


@hydra.main(config_path="config", config_name="net2deeper", version_base="1.1")
def increase_difficulty_four_end(config: Configuration):
    def black_box_ppo_execute(result_processor: ResultProcessor):
        # Mention the used libraries because of implicit imports
        minihack
        gym

        feature_extractor_depth = config.non_hyperparameters.feature_extractor_depth
        seed = config["seed"]
        set_random_seed(seed, using_cuda=True)

        non_hyperparameters = config["non_hyperparameters"]
        environment_name = non_hyperparameters["environment_name"]
        environment_name = environment_name.replace("&", str(budget_level_dict[feature_extractor_depth]))
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
            cnn_intermediate_dimension,
        ) = extract_hyperparameters_minihack(config)
        hyperparameter_str_identifier = str(extract_hyperparameters_minihack(config))

        # Todo rebuild the convert space functionality from stablebaselines to work with a reliable gym env
        # https://github.com/DLR-RM/stable-baselines3/blob/5623d98f9d6bcfd2ab450e850c3f7b090aef5642/stable_baselines3/common/vec_env/patch_gym.py#L63

        # We always use the same seeds in here
        training_vec_env = make_minihack_vec_env(
            environment_name,
            non_hyperparameters["observation_keys"],
            non_hyperparameters["env_seed"],
            non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["max_episode_steps"],
        )

        # Check whether to wrap in monitor wrapper
        evaluation_vec_env = make_minihack_vec_env(
            environment_name,
            non_hyperparameters["observation_keys"],
            non_hyperparameters["env_seed"] + non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["parallel_vec_envs"],
            non_hyperparameters["max_episode_steps"],
        )
        torch.cuda.torch.cuda.empty_cache()

        feature_extractor = partial(
            Net2DeeperFeatureExtractor,
            cnn_intermediate_dimension=cnn_intermediate_dimension,
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
        if feature_extractor_depth > 1:  # If we can load a previous model
            # Rebuild the model with increased size, to then load the weigths and optimizer
            for _ in range(1, feature_extractor_depth - 1):
                model.policy.features_extractor.add_layer()
                additional_layer = model.policy.features_extractor.linear_layers.sequential_container[-2]
                additional_layer.to("cuda")
                model.policy.optimizer.add_param_group({"params": additional_layer.parameters()})

            # Load Previously used model
            final_load_path = get_model_save_path_minihack(config.non_hyperparameters.model_save_path, config, feature_extractor_depth - 1, seed)
            model.set_parameters(os.path.join(final_load_path, "model.zip"), exact_match=False)
            # Add Linear Layer and move to cuda
            model.policy.features_extractor.add_layer()
            model.policy.to("cuda")
            # Add Linear Layer to Optimizer
            additional_layer = model.policy.features_extractor.linear_layers.sequential_container[-2]
            model.policy.optimizer.add_param_group({"params": additional_layer.parameters()})

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
            evaluation_callback.log_losses(result_processor, non_hyperparameters["trial_number"], seed, ent_coef, vf_coef, hyperparameter_str_identifier=hyperparameter_str_identifier)
            evaluation_callback.log_results(result_processor, non_hyperparameters["trial_number"], seed)

        # TODO Save the model and feature extractor
        final_save_path = get_model_save_path_minihack(config.non_hyperparameters.model_save_path, config, feature_extractor_depth, seed)
        if not os.path.exists(final_save_path):
            os.makedirs(final_save_path)

        model.save(os.path.join(final_save_path, "model"))

        evaluation_vec_env = make_minihack_vec_env(
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
        final_score, final_std, actions_per_epiosode = evaluate_policy(
            model,
            evaluation_vec_env,
            n_eval_episodes=non_hyperparameters["n_evaluation_episodes"],
            deterministic=True,
            render=False,
            callback=callback_wrapper.get_callback(),
        )
        if not debug_mode:
            callback_wrapper.process_results(
                non_hyperparameters["trial_number"], seed, final_score, final_std, actions_per_epiosode, budget=feature_extractor_depth, hyperparameter_str_identifier=hyperparameter_str_identifier
            )

            log_results(
                result_processor,
                {
                    "configurations": {
                        "worker_number": seed,  # Currently the same as the workerseed
                        "worker_seed": seed,
                        "trial_number": non_hyperparameters["trial_number"],
                        "budget": feature_extractor_depth,
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
                        "feature_extractor_layer_width": feature_extractor_layer_width,
                        "cnn_intermediate_dimension": cnn_intermediate_dimension,
                        "final_score": final_score,
                        "final_std": final_std,
                    }
                },
            )

        model.policy = None
        torch.cuda.empty_cache()
        return float(-final_score)

    if debug_mode:
        return black_box_ppo_execute(None)
    else:
        # Attach Execution to pyExperimetner to then enable logging
        experimenter = create_pyexperimenter(config)
        result = experimenter.attach(black_box_ppo_execute, config.non_hyperparameters.experiment_id)
        return float(result)


if __name__ == "__main__":
    increase_difficulty_four_end()
