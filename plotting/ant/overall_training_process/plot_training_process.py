import pandas as pd
import seaborn as sns
from utils.plotting import training_process_style, select_incumbents, get_logtable, set_rc_params, get_table
import matplotlib.pyplot as plt
from typing import Dict

set_rc_params()


def get_data(database_name: str, baseline_experiment_ids: Dict[int, int], appraoch_experiment_ids: Dict[int, int]) -> pd.DataFrame:
    all_baseline_data = dict()
    min_cost = float("inf")
    for nettwork_depth, baseline_experiment_id in baseline_experiment_ids.items():
        main_table = get_table(database_name=database_name, table_name="ant_bb_net2deeper_baseline")
        experiment = main_table[main_table["ID"] == baseline_experiment_id]
        max_baseline_timesteps = experiment["total_timesteps"].values[0] * experiment["smac_budget"].values[0]

        baseline_data = get_logtable(database_name=database_name, table_name="ant_bb_net2deeper_baseline", logtable_name="smac_callbacks")
        baseline_data = baseline_data[baseline_data["experiment_id"] == baseline_experiment_id]

        baseline_data = select_incumbents(baseline_data)
        baseline_data["environment_interactions"] = baseline_data["trial_number"] * experiment["total_timesteps"].iloc[0] + experiment["total_timesteps"].iloc[0]
        baseline_data = pd.concat(
            [baseline_data, pd.DataFrame(({"environment_interactions": max_baseline_timesteps, "cost": baseline_data["cost"].min(), "trial_number": 999999999},))],
            ignore_index=True,
        )
        all_baseline_data[nettwork_depth] = baseline_data

    all_appraoch_data = dict()
    for appraoch_name, approach_experiment_id in appraoch_experiment_ids.items():
        main_table = get_table(database_name=database_name, table_name="ant_net2deeper")
        experiment = main_table[main_table["ID"] == approach_experiment_id]
        max_approach_timesteps = experiment["total_timesteps"].values[0] * experiment["smac_budget"].values[0]

        appraoch_data = get_logtable(database_name=database_name, table_name="ant_net2deeper", logtable_name="smac_callbacks")
        appraoch_data = appraoch_data[appraoch_data["experiment_id"] == approach_experiment_id]
        appraoch_data = select_incumbents(appraoch_data)
        appraoch_data["environment_interactions"] = appraoch_data["trial_number"] * experiment["total_timesteps"].iloc[0] + experiment["total_timesteps"].iloc[0]
        appraoch_data = pd.concat(
            [appraoch_data, pd.DataFrame(({"environment_interactions": max_approach_timesteps, "cost": appraoch_data["cost"].min(), "trial_number": 999999999},))],
            ignore_index=True,
        )

        all_appraoch_data[appraoch_name] = appraoch_data

    return all_baseline_data, all_appraoch_data


def plot_one_million_interactions():
    database_name = "fehring_growing_nn_new_seeded"
    baseline_experiment_ids = {"Baseline (2 layers)": 18, "Baseline (4 layers)": 19, "Baseline (8 layers)": 20}

    appraoch_experiment_ids = {"Net2Deeper (4 layers)": 9, "Net2Deeper (8 layers)": 22}

    baseline_data, appraoch_data = get_data(database_name, baseline_experiment_ids, appraoch_experiment_ids)

    training_process_style()
    line_styles = ["-", "--", ":", "-.", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1))]

    line_number = 0
    for approach_name, data in baseline_data.items():
        # The environment interactions column appears to be buggy
        sns.lineplot(x="environment_interactions", y="cost", data=data, label=f"{approach_name}", drawstyle="steps-pre", linestyle=line_styles[line_number])
        line_number += 1

    for approach_name, data in appraoch_data.items():
        sns.lineplot(x="environment_interactions", y="cost", data=data, label=f"{approach_name}", drawstyle="steps-pre", linestyle=line_styles[line_number])

    plt.title("Optimization Process - Ant-v4", fontsize=18, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=14)
    plt.ylabel("Cost", fontsize=14)

    plt.legend(title="Model Type", fontsize=12, title_fontsize=14, loc="center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.ylim(-1800, 200)

    plt.savefig("plotting/ant/overall_training_process/ant_training_process", bbox_inches="tight")


plot_one_million_interactions()
