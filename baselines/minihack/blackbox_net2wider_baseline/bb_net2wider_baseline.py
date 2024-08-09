import gym
import minihack
from ConfigSpace import Configuration
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO
import hydra
from utils import make_minihack_vec_env, extract_hyperparameters_minihack, create_pyexperimenter, log_results, extract_feature_extractor_architecture
from py_experimenter.result_processor import ResultProcessor
from stable_baselines3.common.evaluation import evaluate_policy
from utils.stable_baselines_callback import CustomEvaluationCallback, FinalEvaluationWrapper
from utils.networks.feature_extractor import FixedArchitectureFeatureExtractor
import torch
from functools import partial

debug_mode = False


@hydra.main(config_path="config", config_name="blacbox_net2wider_baseline", version_base="1.1")
def net2wider_baseline(config: Configuration):
    def net2wider_baseline_execute(result_processor: ResultProcessor):
        # Mention the used libraries because of implicit imports
        minihack
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
            feature_extractor_output_dimension,
            _,
            _,
            cnn_intermediate_dimension,
        ) = extract_hyperparameters_minihack(config)

        feature_extractor_architecture = extract_feature_extractor_architecture(config)

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
            FixedArchitectureFeatureExtractor,
            cnn_intermediate_dimension=cnn_intermediate_dimension,
            feature_extractor_architecture=feature_extractor_architecture,
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
                        "feature_extractor_output_dimension": feature_extractor_output_dimension,
                        "final_score": final_score,
                        "final_std": final_std,
                    }
                },
            )
        model.policy = None
        torch.cuda.empty_cache()
        return float(-final_score)

    if debug_mode:
        return net2wider_baseline_execute(None)
    else:
        # Attach Execution to pyExperimetner to then enable logging
        experimenter = create_pyexperimenter(config)
        result = experimenter.attach(net2wider_baseline_execute, config.non_hyperparameters.experiment_id)
        return float(result)


if __name__ == "__main__":
    net2wider_baseline()
