from py_experimenter.result_processor import ResultProcessor
from copy import deepcopy
import numpy as np

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Dict
import os
from typing import Any
import logging


class CustomEvaluationCallback(EvalCallback):
    """
    Builds on standard eval Callback. Idea to track the different losses and percentage of solved episodes additionally.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = list()

        # List of actions and rewards per episode dicts.
        # List entry is evaluation round, first dimension is the vecenv number, second dimension is the episode number
        self.actions_per_episodes = list()
        self.rewards_per_episodes = list()

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
                        "Training and eval env are not wrapped the same way, " "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback " "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, actions_per_episode, rewards_per_episode = evaluate_policy(
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

                # Transform Actions per episode and rewards per episode to numpy array
                # First dimension vecenv_number, second dimension episode_number, third dimension step_number
                self.actions_per_episodes.append(actions_per_episode)
                self.rewards_per_episodes.append(rewards_per_episode)

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

    def log_losses(self, result_processor: ResultProcessor, trial_number: int, worker_number: int, ent_coef: float, vf_coef: float, **kwargs):
        for n_rollout, rollout_losses in enumerate(self.losses):
            rollout_losses = {key[6:]: value for key, value in rollout_losses.items()}
            if "entropy_loss" in rollout_losses:
                rollout_losses["entropy_loss"] = ent_coef * rollout_losses["entropy_loss"]
            if "value_loss" in rollout_losses:
                rollout_losses["value_loss"] = vf_coef * rollout_losses["value_loss"]
            result_processor.process_logs({"training_losses": {"trial_number": trial_number, "worker_number": worker_number, "n_rollout": n_rollout, **rollout_losses, **kwargs}})

    def log_results(self, result_processor: ResultProcessor, trial_number: int, worker_number: int, minihack_adaptation: bool = True, **kwargs):
        try:
            callback_data = np.load("logs/evaluations.npz")

            for timestep, result, _, actions_per_episode, rewards_per_episode in zip(*callback_data.values(), self.actions_per_episodes, self.rewards_per_episodes):
                mean_cost = np.mean(result)
                mean_cost_stdev = np.std(result)
                if minihack_adaptation:
                    result_processor.process_logs(
                        {
                            "training_process": {
                                "worker_id": worker_number,
                                "trial_number": trial_number,
                                "timestep": timestep,
                                "mean_cost": mean_cost,
                                "mean_cost_stdev": mean_cost_stdev,
                                "all_costs": str(result),
                                "actions_per_episode": str(actions_per_episode),
                                "rewards_per_episode": str(rewards_per_episode),
                                **kwargs,
                            }
                        },
                    )
                else:
                    overall_actions = list()
                    for env in range(len(actions_per_episode)):
                        for episode in range(len(actions_per_episode[env])):
                            sum_of_actions = 0
                            for step in range(len(actions_per_episode[env][episode])):
                                sum_of_actions += np.absolute(actions_per_episode[env][episode][step]).sum()
                            overall_actions.append(sum_of_actions / len(actions_per_episode[env][episode]))

                    result_processor.process_logs(
                        {
                            "training_process": {
                                "worker_id": worker_number,
                                "trial_number": trial_number,
                                "timestep": timestep,
                                "mean_cost": mean_cost,
                                "mean_cost_stdev": mean_cost_stdev,
                                "all_costs": str(result),
                                "action_sizes": str(overall_actions),
                                "action_sizes_mean": np.mean(overall_actions),
                                **kwargs,
                            }
                        },
                    )
        except FileNotFoundError as err:
            logging.exception(err)
            return

    def _on_rollout_end(self) -> None:
        return self.losses.append(deepcopy(self.logger.name_to_value))


class FinalEvaluationWrapper:
    """
    Wrapper to evaluate the final results of the model. Adaptation of the standard Stable Baselines3 EvalCallback.
    """

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

    def get_callback(self, minihack_adaptation: bool = True):
        """
        Creates a callback function used for the final evaluation.
        """

        def _minihack_callback(locals_dict, globals_dict, env_number: int):
            end_status = locals_dict["infos"][env_number]["end_status"]
            current_reward = locals_dict["rewards"][env_number]
            self._rewards_in_current_episode[env_number].append(current_reward)
            # Only save data if the state chagned
            if end_status != 0:
                current_step = locals_dict["current_lengths"][env_number]
                # current state == 2 means that the episode finished successfully
                if end_status == 2:
                    self.final_espisode_state.append("Success")
                    self.episode_lengths.append(current_step)
                    self.rewards_per_episode.append(self._rewards_in_current_episode[env_number])
                    self._rewards_in_current_episode[env_number] = []
                # current state == -1 means that the episode failed
                elif end_status == -1:
                    self.final_espisode_state.append("TimeOut")
                    self.episode_lengths.append(current_step)
                    self.rewards_per_episode.append(self._rewards_in_current_episode[env_number])
                    self._rewards_in_current_episode[env_number] = []
                    # current state == 0 means that the episode is currenlty running
                elif end_status == 1:
                    self.final_espisode_state.append("Death")
                    self.episode_lengths.append(current_step)
                    self.rewards_per_episode.append(self._rewards_in_current_episode[env_number])
                    self._rewards_in_current_episode[env_number] = []

                elif end_status not in (-1, 0, 1, 2):
                    raise ValueError(f"Unknown end_status {end_status}")
            self._current_end_status = [locals_dict["infos"][i]["end_status"] for i in range(self.n_envs)]

        def _standard_callback(locals_dict, globals_dict, env_number: int):
            done = locals_dict["dones"][env_number]
            current_reward = locals_dict["rewards"][env_number]
            self._rewards_in_current_episode[env_number].append(current_reward)
            # Only save data if the state chagned
            if done:
                if locals_dict["rewards"][env_number] < 0:
                    self.final_espisode_state.append("Death")
                else:
                    self.final_espisode_state.append("Success")
                self.episode_lengths.append(locals_dict["current_lengths"][env_number])
                self.rewards_per_episode.append(self._rewards_in_current_episode[env_number])
                self._rewards_in_current_episode[env_number] = []

        if minihack_adaptation:
            return _minihack_callback
        else:
            return _standard_callback

    def process_results(self, trial_number: int, worker_number: int, final_score: float, final_std: float, actions_per_episode, **kwargs):
        successfull = np.count_nonzero(np.array(self.final_espisode_state) == "Success") / len(self.final_espisode_state)
        dead = np.count_nonzero(np.array(self.final_espisode_state) == "Death") / len(self.final_espisode_state)
        time_out = np.count_nonzero(np.array(self.final_espisode_state) == "TimeOut") / len(self.final_espisode_state)

        self.result_processor.process_logs(
            {
                "final_evaluation_callback": {
                    **{
                        "trial_number": trial_number,
                        "worker_number": worker_number,
                        "final_score": final_score,
                        "final_std": final_std,
                        "episode_lengths": ",".join([str(x) for x in self.episode_lengths]),
                        "average_episode_lengths": float(np.mean(self.episode_lengths)),
                        "successfull": successfull,
                        "dead": dead,
                        "time_out": time_out,
                        "end_states": ",".join(self.final_espisode_state),
                        "rewards_per_episode": str(self.rewards_per_episode),
                        "actions_per_episode": str(actions_per_episode),
                    },
                    **kwargs,
                }
            }
        )
