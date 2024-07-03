# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import logging
import os
import time
from typing import List

import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter
from hydra.utils import to_absolute_path

# from deepcave import Recorder, Objective
from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier.scheduled_hyperband import ScheduledHyperband
from smac.intensifier.hyperband import Hyperband
from smac.runhistory.dataclasses import TrialValue
from utils.smac_config_selector import LowestBudgetConfigSelector

from py_experimenter.result_processor import ResultProcessor

log = logging.getLogger(__name__)


class HydraSMAC:
    def __init__(
        self,
        global_overrides,
        launcher,
        budget_arg_name,
        save_arg_name,
        n_trials,
        cs,
        result_processor: ResultProcessor,
        seeds=False,
        slurm=False,
        slurm_timeout=10,
        max_parallelization=0.1,
        job_array_size_limit=100,
        intensifier="HB",
        max_budget=None,
        deterministic=True,  # TODO was heisst der deterministic parameter here?
        base_dir=False,
        min_budget=None,
        maximize=False,
    ):
        """ """
        self.global_overrides = global_overrides
        self.launcher = launcher
        self.budget_arg_name = budget_arg_name
        self.save_arg_name = save_arg_name
        self.configspace = cs
        self.output_dir = to_absolute_path(base_dir) if base_dir else to_absolute_path("./")
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.job_idx = 0
        self.seeds = seeds
        if (seeds or not deterministic) and len(self.global_overrides) > 0:
            for i in range(len(self.global_overrides)):
                if self.global_overrides[i].split("=")[0] == "seed":
                    self.global_overrides = self.global_overrides[:i] + self.global_overrides[i + 1 :]
                    break

        self.maximize = maximize
        self.slurm = slurm
        self.slurm_timeout = slurm_timeout
        self.max_parallel = min(job_array_size_limit, max(1, int(max_parallelization * n_trials)))
        self.min_budget = min_budget
        self.iteration = 0
        self.n_trials = n_trials
        self.opt_time = 0
        self.incumbent = []
        self.history = {}
        self.history["configs"] = []
        self.history["performances"] = []
        self.history["budgets"] = []
        self.deterministic = deterministic
        self.max_budget = max_budget

        self.scenario = Scenario(self.configspace, deterministic=deterministic, n_trials=n_trials, min_budget=min_budget, max_budget=max_budget)
        max_config_calls = len(self.seeds) if seeds and not deterministic else 1
        if intensifier == "SHB":
            self.intensifier = ScheduledHyperband(self.scenario, n_lowest_budget=10, bracket_width=max_budget, incumbent_selection="highest_budget", n_seeds=max_config_calls, eta=1.3)
            config_selector = LowestBudgetConfigSelector(self.scenario, min_trials=10)
        elif intensifier == "HB":
            self.intensifier = Hyperband(self.scenario, n_seeds=max_config_calls, eta=2.5)

        else:
            self.intensifier = HyperparameterOptimizationFacade.get_intensifier(
                self.scenario,
                max_config_calls=max_config_calls,
            )
            config_selector = HyperparameterOptimizationFacade.get_config_selector(scenario=self.scenario)

        def dummy(arg, seed, budget):
            pass

        from utils.smac_callbacks import CustomCallback

        self.smac = HyperparameterOptimizationFacade(
            self.scenario,
            dummy,
            callbacks=[CustomCallback(result_processor)],
            intensifier=self.intensifier,
            overwrite=True,
            config_selector=config_selector,
            initial_design=HyperparameterOptimizationFacade.get_initial_design(self.scenario, n_configs=10),
        )

        self.categorical_hps = [n for n in list(self.configspace.keys()) if isinstance(self.configspace.get_hyperparameter(n), CategoricalHyperparameter)]
        self.categorical_hps += [n for n in list(self.configspace.keys()) if isinstance(self.configspace.get_hyperparameter(n), OrdinalHyperparameter)]
        self.continuous_hps = [n for n in list(self.configspace.keys()) if n not in self.categorical_hps]
        self.hp_bounds = np.array(
            [
                [
                    self.configspace.get_hyperparameter(n).lower,
                    self.configspace.get_hyperparameter(n).upper,
                ]
                for n in list(self.configspace.keys())
                if n not in self.categorical_hps
            ]
        )

    def run_configs(self, configs, config_ids: List[int], budgets, seeds, experiment_id: int, trial_numbers: List):
        """
        Run a set of overrides

        Parameters
        ----------
        overrides: List[Tuple]
            A list of overrides to launch

        Returns
        -------
        List[float]
            The resulting performances.
        List[float]
            The incurred costs.
        """
        # Generate overrides
        # TODO: handle budget correctly
        overrides = []

        if self.seeds:  # If we have a seeded environment, each of the seeded runs needs the same
            # Budget
            budgets = np.repeat(budgets, len(self.seeds))

        for i in range(len(configs)):
            names = (
                list(configs[0].keys())
                + ["non_hyperparameters.config_id"]
                + [self.budget_arg_name]
                + [self.save_arg_name]
                + ["non_hyperparameters.experiment_id"]
                + ["seed"]
                + ["non_hyperparameters.trial_number"]
            )  # Add PyExperiemtner ID
            if self.slurm:
                names += ["hydra.launcher.timeout_min"]
                optimized_timeout = self.slurm_timeout * 1 / (self.total_budget // budgets[i]) + 0.1 * self.slurm_timeout

            if self.seeds and self.deterministic:
                for s in self.seeds:
                    save_path = os.path.join(os.getcwd(), "smac3_output", self.scenario.name)
                    values = list(configs[i].values()) + config_ids + [budgets[i]] + [save_path] + [experiment_id] + [s] + [trial_numbers[i]]  # Add PyExperiemtner ID
                    if self.slurm:
                        raise ValueError("Not Supported")
                        values += [int(optimized_timeout)]
                    job_overrides = tuple(self.global_overrides) + tuple(f"{name}={val}" for name, val in zip(names, values))
                    overrides.append(job_overrides)
            elif not self.deterministic:
                raise ValueError("Not Supported")
                save_path = os.path.join(self.checkpoint_dir, f"iteration_{self.iteration}_id_{i}_s{s}.pt")
                values = list(configs[i].values()) + [budgets[i]] + [save_path] + [experiment_id] + [s]  # Add PyExperiemtner ID
                if self.slurm:
                    values += [int(optimized_timeout)]
                job_overrides = tuple(self.global_overrides) + tuple(f"{name}={val}" for name, val in zip(names, values + [seeds[i]]))
                overrides.append(job_overrides)
            else:
                save_path = os.path.join(self.checkpoint_dir, f"iteration_{self.iteration}_id_{i}.pt")
                values = list(configs[i].values()) + config_ids[i] + [budgets[i]] + [save_path] + [experiment_id]  # Add PyExperiemtner ID
                if self.slurm:
                    values += [int(optimized_timeout)]
                job_overrides = tuple(self.global_overrides) + tuple(f"{name}={val}" for name, val in zip(names, values))
                overrides.append(job_overrides)

        # Run overrides
        res = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
        self.job_idx += len(overrides)
        done = False
        while not done:
            for j in range(len(overrides)):
                try:
                    res[j].return_value
                    done = True
                except:
                    done = False

        performances = []  # Performance variable, filled with average value, if seeds used, otherwise fileld with normal value
        costs = []  # Costs: Filled with max cost, if seeds used, standard cost otherwise
        if self.seeds and self.deterministic:
            for j in range(0, len(configs)):
                performances.append(np.mean([res[j * k + k].return_value for k in range(len(self.seeds))]))
                costs.append(int(np.max([budgets[j * k + k] for k in range(len(self.seeds))])))
        else:
            for j in range(len(overrides)):
                performances.append(res[j].return_value)
                costs.append(int(budgets[i]))
        if self.maximize:
            performances = [-p for p in performances]
        return performances, costs

    def get_incumbent(self):
        """
        Get the best sequence of configurations so far.

        Returns
        -------
        List[Configuration]
            Sequence of best hyperparameter configs
        Float
            Best performance value
        """
        best_current_id = np.argmin(self.history["performances"])
        inc_performance = self.history["performances"][best_current_id]
        inc_config = self.history["configs"][best_current_id]
        return inc_config, inc_performance

    def record_iteration(self, performances, configs, budgets):
        """
        Add current iteration to history.

        Parameters
        ----------
        performances: List[float]
            A list of the latest agent performances
        configs: List[Configuration]
            A list of the recent configs
        """
        for i in range(len(configs)):
            self.history["configs"].append(configs[i])
            self.history["performances"].append(performances[i])
            self.history["budgets"].append(budgets[i])
            self.iteration += 1

    def _save_incumbent(self, name=None):
        """
        Log current incumbent to file (as well as some additional info).

        Parameters
        ----------
        name: str | None
            Optional filename
        """
        if name is None:
            name = "incumbent.json"
        res = dict()
        incumbent, inc_performance = self.get_incumbent()
        res["config"] = incumbent.get_dictionary()
        res["score"] = float(inc_performance)
        res["total_training_steps"] = sum(self.history["budgets"])
        res["total_wallclock_time"] = self.start - time.time()
        res["total_optimization_time"] = self.opt_time
        with open(os.path.join(self.output_dir, name), "a+") as f:
            json.dump(res, f)
            f.write("\n")

    def run(self, verbose, experiment_id: int):
        """
        Actual optimization loop.
        In each iteration:
        - get configs (either randomly upon init or through perturbation)
        - run current configs
        - record performances

        Parameters
        ----------
        verbose: bool
            More logging info

        Returns
        -------
        List[Configuration]
            The incumbent configurations.
        """
        if verbose:
            log.info("Starting SMAC Sweep")
        self.start = time.time()
        while self.iteration < self.n_trials:
            opt_time_start = time.time()
            trial_infos = []
            configs = []
            budgets = []
            seeds = []
            trial_numbers = []
            config_ids = []
            for _ in range(self.max_parallel):
                if len(configs) < self.n_trials:
                    info = self.smac.ask()
                    trial_infos.append(info)
                    trial_numbers.append(self.iteration)
                    configs.append(info.config)
                    config_ids.append(info.config.config_id)
                    if info.budget is not None:
                        budgets.append(info.budget)
                    else:
                        budgets.append(self.max_budget)
                    seeds.append(info.seed)
            self.opt_time += time.time() - opt_time_start
            performances, costs = self.run_configs(configs, config_ids, budgets, seeds, experiment_id, trial_numbers)  # Add PyExperimetner ID
            opt_time_start = time.time()
            if self.seeds and self.deterministic:
                seeds = np.zeros(len(performances))
            for info, performance, cost in zip(trial_infos, performances, costs):
                value = TrialValue(cost=float(-performance) if self.maximize else float(performance), time=cost)
                self.smac.tell(info=info, value=value)
            self.record_iteration(performances, configs, budgets)
            if verbose:
                log.info(f"Finished Iteration {self.iteration}!")
                _, inc_performance = self.get_incumbent()
                log.info(f"Current incumbent currently has a performance of {np.round(inc_performance, decimals=2)}.")
            self._save_incumbent()
            self.opt_time += time.time() - opt_time_start
        total_time = time.time() - self.start
        inc_config, inc_performance = self.get_incumbent()
        if verbose:
            log.info(
                f"Finished SMAC Sweep! Total duration was {np.round(total_time, decimals=2)}s, \
                    best agent had a performance of {np.round(inc_performance, decimals=2)}"
            )
            log.info(f"The incumbent configuration is {inc_config}")
        return self.incumbent
