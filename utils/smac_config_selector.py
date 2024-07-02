from smac.main.config_selector import ConfigSelector
from smac.scenario import Scenario
import numpy as np
from typing import Tuple


class LowestBudgetConfigSelector(ConfigSelector):
    def __init__(self, scenario: Scenario, *, retrain_after: int = 8, retries: int = 16, min_trials: int = 1) -> None:
        super().__init__(scenario, retrain_after=retrain_after, retries=retries, min_trials=min_trials)

    def _collect_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collects the data from the runhistory to train the surrogate model. In the case of budgets, the data
        collection strategy is as follows: Looking from highest to lowest budget, return those observations
        that support at least ``self._min_trials`` points.

        If no budgets are used, this is equivalent to returning all observations.
        """
        assert self._runhistory is not None
        assert self._runhistory_encoder is not None

        # If we use a float value as a budget, we want to train the model only on the highest budget
        unique_budgets: set[float] = {run_key.budget for run_key in self._runhistory if run_key.budget is not None}

        min_budget = min(unique_budgets)

        X, Y = self._runhistory_encoder.transform(budget_subset=[min_budget])
        configs_array = self._runhistory_encoder.get_configurations(budget_subset=[min_budget])

        return X, Y, configs_array
