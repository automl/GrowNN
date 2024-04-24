from smac.callback import Callback, MetadataCallback
from py_experimenter.result_processor import ResultProcessor
import json
from smac.main.config_selector import ConfigSelector
from ConfigSpace import Configuration
import json
import platform
from datetime import datetime
import smac
from smac.callback.callback import Callback
from smac.main.smbo import SMBO

from smac.runhistory import TrialInfo, TrialValue


class CustomMetaCallback(MetadataCallback):
    def __init__(self, result_processor: ResultProcessor) -> None:
        super().__init__()
        self.result_processor = result_processor

    def on_start(self, smbo: SMBO) -> None:
        """Called before the optimization starts."""
        super().on_start(smbo)


class CustomCallback(Callback):
    def __init__(self, result_processor: ResultProcessor) -> None:
        super().__init__()
        self.result_processor: ResultProcessor = result_processor

    def on_start(self, smbo: SMBO) -> None:
        """Called before the optimization starts."""
        super().on_start(smbo)

    def on_end(self, smbo: SMBO) -> None:
        """Called after the optimization finished."""
        super().on_end(smbo)

    def on_iteration_start(self, smbo: SMBO) -> None:
        """Called before the next run is sampled."""
        super().on_iteration_start(smbo)

    def on_iteration_end(self, smbo: SMBO) -> None:
        """Called after an iteration ended."""
        super().on_iteration_end(smbo)

    def on_next_configurations_start(self, config_selector: ConfigSelector) -> None:
        """Called before the intensification asks for new configurations. Essentially, this callback is called
        before the surrogate model is trained and before the acquisition function is called.
        """
        super().on_next_configurations_start(config_selector)

    def on_next_configurations_end(self, config_selector: ConfigSelector, config: Configuration) -> None:
        """Called after the intensification asks for new configurations. Essentially, this callback is called
        before the surrogate model is trained and before the acquisition function is called.
        """
        super().on_next_configurations_end(config_selector, config)

    def on_ask_start(self, smbo: SMBO) -> None:
        """Called before the intensifier is asked for the next trial."""
        super().on_ask_start(smbo)

    def on_ask_end(self, smbo: SMBO, info: TrialInfo) -> None:
        """Called after the intensifier is asked for the next trial."""
        super().on_ask_end(smbo, info)

    def on_tell_start(self, smbo: SMBO, info: TrialInfo, value: TrialValue):
        """Called before the stats are updated and the trial is added to the runhistory. Optionally, returns false
        to gracefully stop the optimization.
        """
        super().on_tell_start(smbo, info, value)
        self.result_processor.process_logs({"smac_callbacks": {"trial_number": len(smbo.runhistory), "cost": value.cost}})

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue):
        """Called after the stats are updated and the trial is added to the runhistory. Optionally, returns false
        to gracefully stop the optimization.
        """
        super().on_tell_end(smbo, info, value)
