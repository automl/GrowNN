import pandas as pd
import seaborn as sns
from utils.plotting import training_process_style, select_incumbents, get_logtable, set_rc_params, get_table
import matplotlib.pyplot as plt
from typing import Dict

set_rc_params()


def get_data(database_name: str, baseline_experiment_ids: Dict[int, int], appraoch_experiment_ids: Dict[int, int]) -> pd.DataFrame:
    all_baseline_data = dict()
    for nettwork_depth, baseline_experiment_id in baseline_experiment_ids.items():
        main_table = get_table(database_name=database_name, table_name="increase_difficulty_baseline")
        experiment = main_table[main_table["ID"] == baseline_experiment_id]

        baseline_data = get_logtable(database_name=database_name, table_name="increase_difficulty_baseline", logtable_name="smac_callbacks")
        baseline_data = baseline_data[baseline_data["experiment_id"] == baseline_experiment_id]
        total_timesteps = experiment["total_timesteps"].iloc[0] + experiment["inc_diff_total_timesteps"].iloc[0]
        max_baseline_timesteps = total_timesteps * experiment["smac_budget"].values[0]

        baseline_data = select_incumbents(baseline_data)
        baseline_data["environment_interactions"] = baseline_data["trial_number"] * total_timesteps + total_timesteps

        baseline_data = pd.concat(
            [baseline_data, pd.DataFrame(({"environment_interactions": max_baseline_timesteps, "cost": baseline_data["cost"].min(), "trial_number": 999999999},))],
            ignore_index=True,
        )
        all_baseline_data[nettwork_depth] = baseline_data

    all_appraoch_data = dict()
    for appraoch_name, approach_experiment_id in appraoch_experiment_ids.items():
        main_table = get_table(database_name=database_name, table_name="increase_difficulty_n2d")
        experiment = main_table[main_table["ID"] == approach_experiment_id]
        max_approach_timesteps = experiment["total_timesteps"].values[0] * experiment["smac_budget"].values[0]

        approach_data = get_logtable(database_name=database_name, table_name="increase_difficulty_n2d", logtable_name="smac_callbacks")
        approach_data = approach_data[approach_data["experiment_id"] == approach_experiment_id]
        approach_data = approach_data[approach_data["budget"] == approach_data["budget"].max()]
        approach_data = select_incumbents(approach_data)
        approach_data["environment_interactions"] = approach_data["trial_number"] * experiment["total_timesteps"].iloc[0] + experiment["total_timesteps"].iloc[0]
        approach_data = pd.concat(
            [approach_data, pd.DataFrame(({"environment_interactions": max_approach_timesteps, "cost": approach_data["cost"].min(), "trial_number": 999999999},))],
            ignore_index=True,
        )

        all_appraoch_data[appraoch_name] = approach_data

    return all_baseline_data, all_appraoch_data


def plot_optimization_process():
    database_name = "fehring_growing_nn_new_seeded"
    baseline_experiment_ids = {"Baseline (1 layer); end": 6, "Baseline (1 layer); mid": 9, "Baseline (2 layers); mid": 1, "Baseline (4 layers); mid": 2, "Baseline (4 layers); end": 3}

    appraoch_experiment_ids = {"Net2Deeper (2 Layers); mid": 1, "Net2Deeper (4 Layers); mid": 2, "Net2Deeper (4 Layers); end": 4}

    baseline_data, appraoch_data = get_data(database_name, baseline_experiment_ids, appraoch_experiment_ids)

    training_process_style()

    for approach_name, data in baseline_data.items():
        # The environment interactions column appears to be buggy
        sns.lineplot(x="environment_interactions", y="cost", data=data, label=f"{approach_name}", drawstyle="steps-post", linestyle="solid")

    for approach_name, data in appraoch_data.items():
        sns.lineplot(x="environment_interactions", y="cost", data=data, label=f"{approach_name}", drawstyle="steps-post", linestyle="dashed")

    plt.title("Increase Difficulty", fontsize=20, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=15)
    plt.ylabel("Cost; Negative Mean Evaluation Return", fontsize=15)

    plt.legend(title="Model Type", fontsize=16, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.21), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig("plotting/minihack/difficulty_increases/optimization_process/training_process.png", bbox_inches="tight")


plot_optimization_process()
