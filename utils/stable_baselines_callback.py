from py_experimenter.result_processor import ResultProcessor
from copy import deepcopy
import numpy as np

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Dict
import os
from typing import Any


class CustomEvaluationCallback(EvalCallback):
    """
    Builds on standard eval Callback. Idea to track the different losses and percentage of solved episodes additionally
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = list()

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any], env_id: int) -> None:
        """
        Wrapepr of superclass method without the env_id parameter
        """
        super()._log_success_callback(locals_, globals_)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def log_results(self, result_processor: ResultProcessor, trial_number: int, worker_number: int):
        for n_rollout, rollout_losses in enumerate(self.losses):
            rollout_losses = {key[6:]: value for key, value in rollout_losses.items()}
            result_processor.process_logs(
                {
                    "training_losses": {
                        "trial_number": trial_number,
                        "worker_number": worker_number,
                        "n_rollout": n_rollout,
                        **rollout_losses,
                    }
                }
            )

    def _on_rollout_end(self) -> None:
        return self.losses.append(deepcopy(self.logger.name_to_value))


class FinalEvaluationWrapper:
    def __init__(self, result_processor: ResultProcessor, n_envs: int, n_episodes: int) -> None:
        self.result_processor: ResultProcessor = result_processor
        self.n_envs = n_envs

        # List of tuples, step and final state
        self.final_espisode_state = list()

        # List of final rewards
        self.episode_lengths = list()

        # To track whether or not a state changed
        self._current_end_status = [0] * n_envs

        # Rewards of each episode. First dimension is the episode. Second simension is the reward fer step
        self.rewards_per_episode = []
        # Helper class to track rewards in current episode. Position i gets reset, upon n_env at position i
        # Going into `done`
        self._rewards_in_current_episode = [deepcopy([]) for i in range(n_envs)]

    def get_callback(self):
        """
        Creates a callback function used for the final evaluation
        """

        def _callback(locals_dict, globals_dict, env_number: int):
            end_status = locals_dict["infos"][env_number]["end_status"]
            current_reward = locals_dict["rewards"][env_number]
            self._rewards_in_current_episode[env_number].append(current_reward)
            # Only save data if the state chagned
            if end_status != 0:
                current_step = locals_dict["current_lengths"][env_number]
                # current state == 2 means that the episode finished successfully
                if end_status == 2:
                    self.final_espisode_state.append(1)
                    self.episode_lengths.append(current_step)
                    self.rewards_per_episode.append(self._rewards_in_current_episode[env_number])
                    self._rewards_in_current_episode[env_number] = []
                # current state == -1 means that the episode failed
                elif end_status == -1:
                    self.final_espisode_state.append(0)
                    self.episode_lengths.append(current_step)
                    self.rewards_per_episode.append(self._rewards_in_current_episode[env_number])
                    self._rewards_in_current_episode[env_number] = []
                    # current state == 0 means that the episode is currenlty running
                elif end_status not in (-1, 0, 2):
                    raise ValueError("Unknown state")
            self._current_end_status = [locals_dict["infos"][i]["end_status"] for i in range(self.n_envs)]

        return _callback

    def process_results(self, trial_number: int, worker_number: int, final_score: float, final_std: float):
        self.result_processor.process_logs(
            {
                "final_evaluation_callback": {
                    "trial_number": trial_number,
                    "worker_number": worker_number,
                    "final_score": final_score,
                    "final_std": final_std,
                    "episode_lengths": ",".join([str(x) for x in self.episode_lengths]),
                    "average_episode_lengths": float(np.mean(self.episode_lengths)),
                    "successfull": np.mean(self.final_espisode_state),
                    "rewards_per_episode": str(self.rewards_per_episode),
                }
            }
        )
