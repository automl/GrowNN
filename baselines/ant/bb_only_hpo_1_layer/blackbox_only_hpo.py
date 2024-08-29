import gymnasium as gym
from ConfigSpace import Configuration
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO
import hydra
from utils import make_ant_vec_env, extract_hyperparameters_gymnasium, create_pyexperimenter, log_results
from py_experimenter.result_processor import ResultProcessor
from stable_baselines3.common.evaluation import evaluate_policy
from utils.stable_baselines_callback import FinalEvaluationWrapper, CustomEvaluationCallback
from utils.gymnasium_compatible.feature_extractor import FixedArchitectureFeaturesExtractor
import torch
from functools import partial
import os
from utils.hyperparameter_handling import extract_feature_extractor_architecture

debug_mode = False


@hydra.main(config_path="config", config_name="blackbox_only_hpo", version_base="1.1")
def ant_bb_1(config: Configuration):
    def black_box_ppo_execute(result_processor: ResultProcessor):
        # Mention the used libraries because of implicit imports
        gym

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

        # We always use the same seeds in here
        training_vec_env = make_ant_vec_env(env_id=environment_name, environment_seed=non_hyperparameters["env_seed"], parralel_vec_envs=non_hyperparameters["parallel_vec_envs"])

        # Check whether to wrap in monitor wrapper
        evaluation_vec_env = make_ant_vec_env(
            env_id=environment_name,
            environment_seed=non_hyperparameters["env_seed"] + non_hyperparameters["parallel_vec_envs"],
            parralel_vec_envs=non_hyperparameters["parallel_vec_envs"],
        )
        torch.cuda.torch.cuda.empty_cache()

        feature_extractor_architecture = extract_feature_extractor_architecture(config)
        pi_dimension = non_hyperparameters["pi_dimension"]
        vf_dimension = non_hyperparameters["vf_dimension"]

        feature_extractor = partial(FixedArchitectureFeaturesExtractor, feature_extractor_architecture=feature_extractor_architecture)
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
            gamma=gamma,
            policy_kwargs={"features_extractor_class": feature_extractor, "net_arch": {"pi": [pi_dimension], "vf": [vf_dimension]}},
        )

        evaluation_callback = CustomEvaluationCallback(
            evaluation_vec_env,
            n_eval_episodes=non_hyperparameters["n_evaluation_episodes"],
            eval_freq=non_hyperparameters["total_timesteps"] / non_hyperparameters["n_evaluation_rounds"] / non_hyperparameters["parallel_vec_envs"],
            deterministic=True,
            render=False,
            log_path="./logs",
        )
        # For Soem Reason the policynet has a input dimension of 1
        try:
            training_diverged = False
            model.learn(total_timesteps=non_hyperparameters["total_timesteps"], callback=evaluation_callback)
        except ValueError:
            training_diverged = True
        if not debug_mode:
            evaluation_callback.log_losses(result_processor, non_hyperparameters["trial_number"], seed, ent_coef, vf_coef)
            evaluation_callback.log_results(result_processor, non_hyperparameters["trial_number"], seed, minihack_adaptation=False)

        if training_diverged:
            return 3000

        evaluation_vec_env = make_ant_vec_env(
            env_id=environment_name,
            environment_seed=non_hyperparameters["env_seed"] + non_hyperparameters["parallel_vec_envs"],
            parralel_vec_envs=non_hyperparameters["parallel_vec_envs"],
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

        callback_wrapper.process_results(non_hyperparameters["trial_number"], seed, final_score, final_std, actions_per_epiosode)
        if not debug_mode:
            log_results(
                result_processor,
                {
                    "configurations": {
                        "worker_number": seed,  # Currently the same as the workerseed
                        "worker_seed": seed,
                        "trial_number": non_hyperparameters["trial_number"],
                        "environment_id": environment_name,
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
                        "feature_extractor_architecture": str(feature_extractor_architecture),
                        "vf_dimension": vf_dimension,
                        "pi_dimension": pi_dimension,
                        "final_score": final_score,
                        "final_std": final_std,
                    }
                },
            )

        model.save(os.path.join(non_hyperparameters["model_save_path"], str(non_hyperparameters["trial_number"]), str(seed), "model"))

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
    ant_bb_1()
