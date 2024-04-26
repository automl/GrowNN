from py_experimenter.result_processor import ResultProcessor
from copy import deepcopy
import numpy as np


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
