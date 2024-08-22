import gym
from ConfigSpace import Configuration
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO
import hydra
from utils import extract_hyperparameters_gymnasium, create_pyexperimenter, log_results, make_ant_vec_env
from py_experimenter.result_processor import ResultProcessor
from stable_baselines3.common.evaluation import evaluate_policy
from utils.stable_baselines_callback import FinalEvaluationWrapper, CustomEvaluationCallback
from utils.gymnasium_compatible.feature_extractor import Net2WiderFeatureExtractor
from utils import get_model_save_path_gymnasium
import torch
from functools import partial
import os

debug_mode = False


@hydra.main(config_path="config", config_name="ant_n2d", version_base="1.1")
def black_box_ppo_configure(config: Configuration):
    def black_box_ppo_execute(result_processor: ResultProcessor):
        # Mention the used libraries because of implicit imports
        gym

        grow_width_budget = config.non_hyperparameters.grow_width_budget
        environment_name = config.non_hyperparameters.environment_id

        # We only seed the neural network. Everything else is seeded more or less constantly
        seed = config["seed"]
        set_random_seed(seed, using_cuda=True)

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
            gamma,
        ) = extract_hyperparameters_gymnasium(config)

        input_size = ...
        output_size = ...
        n_layers = ...
        increase_factor = ...
        noise_level = ...

        # We always use the same seeds in here
        training_vec_env = make_ant_vec_env(env_id=environment_name, environment_seed=non_hyperparameters["env_seed"], parralel_vec_envs=non_hyperparameters["parallel_vec_envs"], hardcore=True)
        # Check whether to wrap in monitor wrapper
        evaluation_vec_env = make_ant_vec_env(
            env_id=environment_name,
            environment_seed=non_hyperparameters["env_seed"] + non_hyperparameters["parallel_vec_envs"],
            parralel_vec_envs=non_hyperparameters["parallel_vec_envs"],
            hardcore=True,
        )
        torch.cuda.torch.cuda.empty_cache()
        feature_extractor = partial(
            Net2WiderFeatureExtractor,
            obseravtion_space=training_vec_env.observation_space,
            input_size=input_size,
            output_size=output_size,
            n_layers=n_layers,
            increase_factor=increase_factor,
            noise_level=noise_level,
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
            gamma=gamma,
            n_steps=n_steps,  # The number of steps to run for each environment per update
            seed=seed,
            policy_kwargs={"features_extractor_class": feature_extractor, "net_arch": {"pi": [non_hyperparameters["pi_dimension"]], "vf": [non_hyperparameters["vf_dimension"]]}},
        )

        if grow_width_budget > 1:  # If we can load a previous model
            for _ in range(1, grow_width_budget - 1):
                model.policy.features_extractor.add_layer()
                additional_layer = model.policy.features_extractor.net2deeper_network.sequential_container[-2]
                additional_layer.to("cuda")
                model.policy.optimizer.add_param_group({"params": additional_layer.parameters()})

            # Load Previously used model
            final_load_path = get_model_save_path_gymnasium(config.non_hyperparameters.model_save_path, config, grow_width_budget - 1, seed)
            model.set_parameters(os.path.join(final_load_path, "model.zip"), exact_match=False)
            # Add Linear Layer and move to cuda
            model.policy.features_extractor.add_layer()
            model.policy.to("cuda")
            # Add Linear Layer to Optimizer
            additional_layer = model.policy.features_extractor.net2deeper_network.sequential_container[-2]
            model.policy.optimizer.add_param_group({"params": additional_layer.parameters()})

        evaluation_callback = CustomEvaluationCallback(
            evaluation_vec_env,
            n_eval_episodes=non_hyperparameters["n_evaluation_episodes"],
            eval_freq=non_hyperparameters["total_timesteps"] / non_hyperparameters["n_evaluation_rounds"] / non_hyperparameters["parallel_vec_envs"],
            deterministic=True,
            render=False,
            log_path="./logs",
        )
        # For Soem Reason the policynet has a input dimension of 1
        model.learn(total_timesteps=non_hyperparameters["total_timesteps"], callback=evaluation_callback)
        if not debug_mode:
            evaluation_callback.log_losses(result_processor, non_hyperparameters["trial_number"], seed, ent_coef, vf_coef)
            evaluation_callback.log_results(result_processor, non_hyperparameters["trial_number"], seed)

        evaluation_vec_env = make_ant_vec_env(
            env_id=environment_name,
            environment_seed=non_hyperparameters["env_seed"] + non_hyperparameters["parallel_vec_envs"],
            parralel_vec_envs=non_hyperparameters["parallel_vec_envs"],
            hardcore=True,
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
            callback=callback_wrapper.get_callback(minihack_adaptation=False),
        )

        callback_wrapper.process_results(
            non_hyperparameters["trial_number"],
            seed,
            final_score,
            final_std,
            actions_per_epiosode,
            budget=grow_width_budget,
            hyperparameter_str_indentifier=str(extract_hyperparameters_gymnasium(config)),
        )
        if not debug_mode:
            log_results(
                result_processor,
                {
                    "configurations": {
                        "worker_number": seed,  # Currently the same as the workerseed
                        "worker_seed": seed,
                        "trial_number": non_hyperparameters["trial_number"],
                        "environment_id": environment_name,
                        "budget": grow_width_budget,
                        "hyperparameter_str_identifier": str(extract_hyperparameters_gymnasium(config)),
                        "gamma": gamma,
                        "feature_extractor_output_dimension": non_hyperparameters["vf_dimension"],
                        "feature_extractor_layer_width": non_hyperparameters["feature_extractor_width"],
                        "n_feature_extractor_layers": n_layers,
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

        final_save_path = get_model_save_path_gymnasium(config.non_hyperparameters.model_save_path, config, grow_width_budget, seed)
        if not os.path.exists(final_save_path):
            os.makedirs(final_save_path)

        model.save(os.path.join(final_save_path, "model"))

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
    black_box_ppo_configure()
