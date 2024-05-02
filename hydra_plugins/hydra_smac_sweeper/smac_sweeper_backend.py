# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import annotations

import logging

from typing import List
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
import tempfile
import os
import operator
from functools import reduce
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import get_class
from hydra_plugins.hydra_smac_sweeper.hydra_smac import HydraSMAC
from hydra_plugins.utils.search_space_encoding import search_space_to_config_space
from utils.py_experimenter_utils import create_pyexperimenter
from omegaconf import DictConfig, OmegaConf, open_dict
from rich import print as printr

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("get_class", get_class, replace=True)


class SMACSweeperBackend(Sweeper):
    def __init__(
        self,
        search_space: DictConfig,
        resume: str | None = None,
        optimizer: str | None = "pbt",
        budget: int | None = None,
        budget_variable: str | None = None,
        loading_variable: str | None = None,
        saving_variable: str | None = None,
        smac_kwargs: DictConfig | dict = {},
    ) -> None:
        """
        Backend for the PBT sweeper.
        Instantiate the sweeper with hydra and launch optimization.

        Parameters
        ----------
        search_space: DictConfig
            The search space, either a DictConfig from a hydra yaml config file,
            or a path to a json configuration space file in the format required of ConfigSpace,
            or already a ConfigurationSpace config space.
        optimizer: str
            Name of the acquisition function boil should use
        budget: int | None
            Total budget for a single population member.
            This could be e.g. the total number of steps to train a single agent.
        budget_variable: str | None
            Name of the argument controlling the budget, e.g. num_steps.
        loading_variable: str | None
            Name of the argument controlling the loading of agent parameters.
        saving_variable: str | None
            Name of the argument controlling the checkpointing.
        pbt_kwargs: DictConfig | None
            Additional PBT specific arguments. These differ between different versions of PBT.
        Returns
        -------
        None

        """
        self.search_space = search_space
        self.optimizer = optimizer
        self.budget_variable = budget_variable
        self.loading_variable = loading_variable
        self.saving_variable = saving_variable
        self.smac_kwargs = smac_kwargs
        self.budget = int(budget)
        self.resume = resume

        self.task_function: TaskFunction | None = None
        self.sweep_dir: str | None = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """
        Setup launcher.

        Parameters
        ----------
        hydra_context: HydraContext
        task_function: TaskFunction
        config: DictConfigf

        Returns
        -------
        None

        """
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(config=config, hydra_context=hydra_context, task_function=task_function)
        self.task_function = task_function
        self.sweep_dir = config.hydra.sweep.dir

    def sweep(self, arguments: List[str]) -> List | None:
        """
        Run PBT optimization and returns the incumbent configurations.

        Parameters
        ----------
        arguments: List[str]
            Hydra overrides for the sweep.

        Returns
        -------
        List[Configuration] | None
            Incumbent (best) configuration.

        """
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        def _sweep(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
            smac = HydraSMAC(
                global_overrides=arguments,
                launcher=self.launcher,
                budget_arg_name=self.budget_variable,
                save_arg_name=self.saving_variable,
                n_trials=self.budget,
                base_dir=self.sweep_dir,
                cs=configspace,
                result_processor=result_processor,
                **self.smac_kwargs,
            )

            incumbent = smac.run(True, result_processor.experiment_id)
            final_config = self.config
            with open_dict(final_config):
                del final_config["hydra"]
            for a in arguments:
                n, v = a.split("=")
                key_parts = n.split(".")
                reduce(operator.getitem, key_parts[:-1], final_config)[key_parts[-1]] = v
            schedules = {}
            for i in range(len(incumbent)):
                for k, v in incumbent[i].items():
                    if k not in schedules.keys():
                        schedules[k] = []
                    schedules[k].append(v)
            for k in schedules.keys():
                key_parts = k.split(".")
                reduce(operator.getitem, key_parts[:-1], final_config)[key_parts[-1]] = schedules[k]
            with open(os.path.join(smac.output_dir, "final_config.yaml"), "w+") as fp:
                OmegaConf.save(config=final_config, f=fp)
            result_processor.process_results(
                {
                    "final_cost": incumbent["cost"],
                    "final_config": final_config,
                }
            )

        printr("Config", self.config)
        printr("Hydra context", self.hydra_context)

        self.launcher.global_overrides = arguments
        if len(arguments) == 0:
            log.info("Sweep doesn't override default config.")
        else:
            log.info(f"Sweep overrides: {' '.join(arguments)}")

        configspace = search_space_to_config_space(search_space=self.search_space)

        py_experimenter = create_pyexperimenter(self.config, use_ssh_tunnel=True)

        keyfield_values = dict(self.config["non_hyperparameters"])

        # Dont write PyExperimetner Experimentid
        del keyfield_values["experiment_id"]
        del keyfield_values["trial_number"]

        keyfield_values["observation_keys"] = ",".join(keyfield_values["observation_keys"])
        py_experimenter.add_experiment_and_execute(keyfield_values, experiment_function=_sweep)
