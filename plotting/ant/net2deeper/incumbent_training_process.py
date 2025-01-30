from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
from utils.plotting import (
    training_process_style,
    get_logtable,
    set_rc_params,
    convert_dataframe_gym,
    format_func,
)
import matplotlib.pyplot as plt
from typing import Dict

set_rc_params()


def get_data(
    database_name: str,
    baseline_experiment_ids: Dict[int, int],
    appraoch_experiment_ids: Dict[int, int],
) -> pd.DataFrame:
    all_baseline_data = dict()
    baseline_training_process_data = get_logtable(
        database_name=database_name,
        table_name="ant_bb_net2deeper_baseline",
        logtable_name="training_process",
    )
    for nettwork_depth, baseline_experiment_id in baseline_experiment_ids.items():
        baseline_smac_callbacks = get_logtable(
            database_name=database_name,
            table_name="ant_bb_net2deeper_baseline",
            logtable_name="smac_callbacks",
        )
        baseline_smac_callbacks = baseline_smac_callbacks[baseline_smac_callbacks["experiment_id"] == baseline_experiment_id]
        incumbent_trial_number = baseline_smac_callbacks[baseline_smac_callbacks["cost"] == baseline_smac_callbacks["cost"].min()]["trial_number"].iloc[0] - 1

        training_process_data = baseline_training_process_data[
            (baseline_training_process_data["trial_number"] == incumbent_trial_number) & (baseline_training_process_data["experiment_id"] == baseline_experiment_id)
        ]
        training_process_data = training_process_data[["timestep", "worker_id", "all_costs"]]
        training_process_data = convert_dataframe_gym(training_process_data)
        all_baseline_data[nettwork_depth] = training_process_data

    all_appraoch_data = dict()
    appraoch_training_process_data = get_logtable(
        database_name=database_name,
        table_name="ant_net2deeper",
        logtable_name="training_process",
    )
    for appraoch_name, approach_experiment_id in appraoch_experiment_ids.items():
        appraoch_smac_callback_data = get_logtable(
            database_name=database_name,
            table_name="ant_net2deeper",
            logtable_name="smac_callbacks",
        )
        appraoch_smac_callback_data = appraoch_smac_callback_data[appraoch_smac_callback_data["experiment_id"] == approach_experiment_id]
        incumbent_hyperparameter_str_identifier = appraoch_smac_callback_data[appraoch_smac_callback_data["cost"] == appraoch_smac_callback_data["cost"].min()]["hyperparameter_str_identifier"].iloc[0]
        trial_numbers = appraoch_smac_callback_data[appraoch_smac_callback_data["hyperparameter_str_identifier"] == incumbent_hyperparameter_str_identifier]["trial_number"] - 1.0

        experiment_training_process_data = appraoch_training_process_data[appraoch_training_process_data["experiment_id"] == approach_experiment_id]
        experiment_training_process_data = {trial_number: experiment_training_process_data[appraoch_training_process_data["trial_number"] == trial_number] for trial_number in trial_numbers}

        max_timestep = experiment_training_process_data[trial_numbers.max()]["timestep"].max()
        for budget, (trial_number, training_process_data) in enumerate(sorted(experiment_training_process_data.items())):
            training_process_data["timestep"] += max_timestep * budget

        approach_data = pd.concat(experiment_training_process_data.values(), ignore_index=True)
        approach_data = approach_data[["timestep", "worker_id", "all_costs"]]
        approach_data = convert_dataframe_gym(approach_data)
        all_appraoch_data[appraoch_name] = approach_data

    return all_baseline_data, all_appraoch_data


def plot_incumbent_interactions_4_layers():
    database_name = "fehring_growing_nn_new_seeded"
    baseline_experiment_ids = {
        "Static (4 layers)": 19,
    }

    appraoch_experiment_ids = {
        "Net2Deeper (4 layers)": 9,
    }

    baseline_data, appraoch_data = get_data(database_name, baseline_experiment_ids, appraoch_experiment_ids)

    training_process_style()

    line_number = 0
    for approach_name, data in baseline_data.items():
        # The environment interactions column appears to be buggy
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="solid",
        )

    for approach_name, data in appraoch_data.items():
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="dashed",
        )
        line_number += 1

    for timestep in range(500000, 2000000, 500000):
        plt.axvline(x=timestep, color="black", linestyle="--", alpha=0.5)

    plt.title("Optimization Process - Ant-v4", fontsize=20, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=14)
    plt.ylabel("IQM of Evaluation Episode Returns", fontsize=14)

    plt.legend(
        title="Model Type",
        fontsize=12,
        title_fontsize=18,
        loc="center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(
        "plotting/ant/net2deeper/incumbent_training_process_4_layers",
        bbox_inches="tight",
    )


def plot_incumbent_interactions_8_layers():
    database_name = "fehring_growing_nn_new_seeded"
    baseline_experiment_ids = {
        "Static (8 layers)": 20,
    }

    appraoch_experiment_ids = {
        "Net2Deeper (8 layers)": 22,
    }

    baseline_data, appraoch_data = get_data(database_name, baseline_experiment_ids, appraoch_experiment_ids)

    training_process_style()

    line_number = 0
    for approach_name, data in baseline_data.items():
        # The environment interactions column appears to be buggy
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="solid",
        )

    for approach_name, data in appraoch_data.items():
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="dashdot",
        )
        line_number += 1

    for timestep in range(250000, 2000000, 250000):
        plt.axvline(x=timestep, color="black", linestyle="--", alpha=0.5)

    plt.title("Optimization Process - Ant-v4", fontsize=20, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=15)
    plt.ylabel("IQM of Evaluation Episode Returns", fontsize=15)

    plt.legend(
        title="Model Type",
        fontsize=16,
        title_fontsize=18,
        loc="center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(
        "plotting/ant/net2deeper/incumbent_training_process_8_layers",
        bbox_inches="tight",
    )


def plot_both_in_one():
    database_name = "fehring_growing_nn_new_seeded"
    baseline_experiment_ids = {
        "Static (5 layers)": 19,
    }

    appraoch_experiment_ids = {
        "GrowNN (5 layers)": 9,
    }

    baseline_data, appraoch_data = get_data(database_name, baseline_experiment_ids, appraoch_experiment_ids)

    training_process_style()

    line_number = 0
    for approach_name, data in baseline_data.items():
        # The environment interactions column appears to be buggy
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="solid",
        )

    for approach_name, data in appraoch_data.items():
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="dashed",
        )
        line_number += 1

    for timestep in range(500000, 2000000, 500000):
        plt.axvline(x=timestep, color="black", linestyle="--", alpha=0.5)

    database_name = "fehring_growing_nn_new_seeded"
    baseline_experiment_ids = {
        "Static (9 layers)": 20,
    }

    appraoch_experiment_ids = {
        "GrowNN (9 layers)": 22,
    }

    baseline_data, appraoch_data = get_data(database_name, baseline_experiment_ids, appraoch_experiment_ids)

    line_number = 0
    for approach_name, data in baseline_data.items():
        # The environment interactions column appears to be buggy
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="solid",
        )

    for approach_name, data in appraoch_data.items():
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="dashdot",
        )
        line_number += 1

    for timestep in range(250000, 2000000, 250000):
        plt.axvline(x=timestep, color="black", linestyle="--", alpha=0.5)

    plt.xlabel("Environment Interactions", fontsize=25)
    plt.yticks([0, 500, 1000, 1500, 2000])
    plt.ylabel("IQM of Evaluation Episode Returns", fontsize=25)

    plt.legend(
        title="Model Type",
        fontsize=28,
        title_fontsize=28,
        loc="center",
        bbox_to_anchor=(0.45, -0.31),
        ncol=2,
    )
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig("plotting/ant/net2deeper/both_in_one.pdf", bbox_inches="tight", pad_inches=0.3)


def subplots_of_both_figures():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(hspace=0.5)

    database_name = "fehring_growing_nn_new_seeded"
    baseline_experiment_ids = {
        "Static (4 layers)": 19,
    }

    appraoch_experiment_ids = {
        "Net2Deeper (4 layers)": 9,
    }

    baseline_data, appraoch_data = get_data(database_name, baseline_experiment_ids, appraoch_experiment_ids)

    # 4 layers
    line_number = 0
    for approach_name, data in baseline_data.items():
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="solid",
            ax=axs[0],
        )

    for approach_name, data in appraoch_data.items():
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="dashed",
            ax=axs[0],
        )
        line_number += 1

    for timestep in range(500000, 2000000, 500000):
        axs[0].axvline(x=timestep, color="black", linestyle="--", alpha=0.5)

    axs[0].set_title("Optimization Process - Ant-v4 (4 layers)", fontsize=14, fontweight="bold")

    axs[0].set_xlabel("Environment Interactions", fontsize=10)
    axs[0].set_ylabel("Cost", fontsize=10)

    axs[0].legend(
        title="Model Type",
        fontsize=8,
        title_fontsize=10,
        loc="center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].xaxis.set_major_formatter(FuncFormatter(format_func))

    # 8 layers
    baseline_experiment_ids = {
        "Static (8 layers)": 20,
    }
    appraoch_experiment_ids = {
        "Net2Deeper (8 layers)": 22,
    }

    line_number = 0
    for approach_name, data in baseline_data.items():
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="solid",
            ax=axs[1],
        )

    for approach_name, data in appraoch_data.items():
        sns.lineplot(
            x="timestep",
            y="episode_reward",
            data=data,
            label=f"{approach_name}",
            linestyle="dashdot",
            ax=axs[1],
        )
        line_number += 1

    for timestep in range(250000, 2000000, 250000):
        axs[1].axvline(x=timestep, color="black", linestyle="--", alpha=0.5)

    axs[1].set_title("Optimization Process - Ant-v4 (8 layers)", fontsize=14, fontweight="bold")
    axs[1].set_xlabel("Environment Interactions", fontsize=10)
    axs[1].set_ylabel("Cost; Negative Mean Evaluation Return", fontsize=10)

    axs[1].legend(
        title="Model Type",
        fontsize=8,
        title_fontsize=10,
        loc="center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )
    axs[1].grid(True, linestyle="--", alpha=0.7)
    axs[1].xaxis.set_major_formatter(FuncFormatter(format_func))

    fig.suptitle("Optimization Process - Ant-v4", fontsize=16, fontweight="bold")
    fig.savefig("plotting/ant/net2deeper/incumbent_training_process_both.pdf", bbox_inches="tight")


# plot_incumbent_interactions_4_layers()
# plot_incumbent_interactions_8_layers()
plot_both_in_one()
# subplots_of_both_figures()
