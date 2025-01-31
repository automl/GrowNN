import pandas as pd
import seaborn as sns
from utils.plotting import training_process_style, get_logtable, set_rc_params, convert_dataframe_gym
import matplotlib.pyplot as plt
from typing import Dict

set_rc_params()


def get_data(database_name: str, appraoch_experiment_ids: Dict[int, int]) -> pd.DataFrame:
    appraoch_training_process_data = get_logtable(database_name=database_name, table_name="ant_net2deeper", logtable_name="training_process")
    for appraoch_name, approach_experiment_id in appraoch_experiment_ids.items():
        appraoch_smac_callback_data = get_logtable(database_name=database_name, table_name="ant_net2deeper", logtable_name="smac_callbacks")
        appraoch_smac_callback_data = appraoch_smac_callback_data[appraoch_smac_callback_data["experiment_id"] == approach_experiment_id]
        incumbent_hyperparameter_str_identifier = appraoch_smac_callback_data[appraoch_smac_callback_data["cost"] == appraoch_smac_callback_data["cost"].min()]["hyperparameter_str_identifier"].iloc[0]
        trial_numbers = appraoch_smac_callback_data[appraoch_smac_callback_data["hyperparameter_str_identifier"] == incumbent_hyperparameter_str_identifier]["trial_number"] - 1.0

        experiment_training_process_data = appraoch_training_process_data[appraoch_training_process_data["experiment_id"] == approach_experiment_id]
        experiment_training_process_data = {trial_number: experiment_training_process_data[appraoch_training_process_data["trial_number"] == trial_number] for trial_number in trial_numbers}

        max_timestep = experiment_training_process_data[trial_numbers.max()]["timestep"].max()
        for budget, (trial_number, training_process_data) in enumerate(sorted(experiment_training_process_data.items())):
            training_process_data["timestep"] += max_timestep * budget

        approach_data = pd.concat(experiment_training_process_data.values(), ignore_index=True)
        action_data = approach_data[["timestep", "action_sizes_mean"]]
        approach_data = approach_data[["timestep", "worker_id", "all_costs"]]
        approach_data = convert_dataframe_gym(approach_data)

        return (approach_data, action_data)


def plot_incumbent_interactions_8_layers():
    database_name = "fehring_growing_nn_new_seeded"

    appraoch_experiment_ids = {
        "Net2Deeper (8 layers)": 22,
    }

    training_process_data, action_data = get_data(database_name, appraoch_experiment_ids)

    training_process_style()

    sns.lineplot(x="timestep", y="episode_reward", data=training_process_data, label="Net2Deeper (8 layers)", linestyle="dashed")
    plt.ylabel("IQM of Evaluation Epsiode Return", fontsize=15)
    # get ax object

    action_size_color = sns.color_palette()[1]
    legend_labels = ["Net2Deeper (8 layers)", "Average Action Size"]
    # second value in the color palette
    legend_items = [plt.Line2D([0], [0], color="blue", linestyle="--"), plt.Line2D([0], [0], color=action_size_color, linestyle=":")]
    plt.legend(legend_items, legend_labels, fontsize=16, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.grid(True, linestyle="--", alpha=0.7)
    ax = plt.gca()
    plt.xlabel("Environment Interactions", fontsize=15)
    ax2 = ax.twinx()
    sns.lineplot(x="timestep", y="action_sizes_mean", data=action_data, color=action_size_color, ax=ax2, linestyle="-")

    for timestep in range(250000, 2000000, 250000):
        plt.axvline(x=timestep, color="black", linestyle="--", alpha=0.5)

    plt.title("Optimization Process - Ant-v4", fontsize=20, fontweight="bold")

    plt.ylabel("Average Action Size", fontsize=15)
    ax2.grid(False)

    plt.savefig("plotting/ant/net2deeper/action_size", bbox_inches="tight")


plot_incumbent_interactions_8_layers()
