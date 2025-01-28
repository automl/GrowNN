from smac.intensifier.hyperband import Hyperband
import math
from typing import Dict
from smac.scenario import Scenario


class ScheduledHyperband(Hyperband):
    def __init__(self, scenario: Scenario, n_lowest_budget: int, bracket_width: int, **kwargs):
        super().__init__(scenario, **kwargs)
        self._n_lowest_budget = n_lowest_budget
        self._bracket_width = bracket_width

    def __post_init__(self) -> None:
        super().__post_init__()

        min_budget = self._min_budget
        max_budget = self._max_budget
        assert min_budget is not None and max_budget is not None

        self._n_configs_in_stage: Dict[int, list] = self.get_n_configs()
        self._budgets_in_stage: Dict[int, list] = self.get_budgets()
        self._max_iterations: Dict[int, int] = self.get_max_iterations()

    def get_budgets(self):
        return {0: list(range(self._min_budget, self._max_budget + 1))}

    def get_n_configs(self):
        return {0: [math.floor(self._n_lowest_budget / (self._eta**i)) for i in range(self._s_max)]}

    def _get_max_iterations(self, *args, **kwargs):
        return self._max_budget - self._min_budget + 1

    def get_max_iterations(self):
        return {0: self._get_max_iterations()}

    @property
    def n_configs(self):
        return len(self.get_n_configs()[0])

    def _get_next_bracket(self) -> int:
        return 0
