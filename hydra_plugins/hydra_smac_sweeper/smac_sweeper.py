# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig


class SMACSweeper(Sweeper):
    """This is basically just a wrapper class for the backend functionality"""

    def __init__(self, *args: Any, **kwargs: Dict[Any, Any]) -> None:
        from .smac_sweeper_backend import SMACSweeperBackend

        self.sweeper = SMACSweeperBackend(*args, **kwargs)

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.sweeper.setup(hydra_context=hydra_context, task_function=task_function, config=config)

    def sweep(self, arguments: List[str]) -> None:
        return self.sweeper.sweep(arguments)
