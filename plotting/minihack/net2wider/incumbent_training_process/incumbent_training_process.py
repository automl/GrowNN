from utils.plotting import get_logtable, set_rc_params, convert_dataframe
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from utils.plotting import training_process_style
from typing import List

set_rc_params()


def get_data(database_name: str, experiment_ids: List[int]):
    database_name = "fehring_growing_nn_new_seeded"
    baseline_width_1_callback_data = get_logtable(database_name=database_name, table_name="incumbent_gen_2_layers", logtable_name="smac_callbacks")
    baseline_width_1_training_process_data = get_logtable(database_name=database_name, table_name="incumbent_gen_2_layers", logtable_name="training_process")

    baseline_width_2_callback_data = get_logtable(database_name=database_name, table_name="bb_net2wider_baseline", logtable_name="smac_callbacks")
    baseline_width_2_training_process_data = get_logtable(database_name=database_name, table_name="bb_net2wider_baseline", logtable_name="training_process")

    baseline_width_4_callback_data = get_logtable(database_name=database_name, table_name="bb_net2wider_baseline", logtable_name="smac_callbacks")
    baseline_width_4_training_process_data = get_logtable(database_name=database_name, table_name="bb_net2wider_baseline", logtable_name="training_process")

    net2wider_smac_callback_data = get_logtable(database_name=database_name, table_name="net2wider_budget200", logtable_name="smac_callbacks")
    net2wider_smac_callback_data = net2wider_smac_callback_data[net2wider_smac_callback_data["trial_number"] <= 182]
    net2wider_smac_callback_data = net2wider_smac_callback_data.reset_index(drop=True)
    net2wider_training_process_data = get_logtable(database_name=database_name, table_name="net2wider_budget200", logtable_name="training_process")

    baseline_width_1_callback_data_relevant = baseline_width_1_callback_data[baseline_width_1_callback_data["experiment_id"] == experiment_ids[0]]
    baseline_width_1_training_process_data = baseline_width_1_training_process_data[baseline_width_1_training_process_data["experiment_id"] == experiment_ids[0]]

    baseline_width_2_callback_data_relevant = baseline_width_2_callback_data[baseline_width_2_callback_data["experiment_id"] == experiment_ids[1]]
    baseline_width_2_training_process_data = baseline_width_2_training_process_data[baseline_width_2_training_process_data["experiment_id"] == experiment_ids[1]]

    baseline_width_4_callback_data_relevant = baseline_width_4_callback_data[baseline_width_4_callback_data["experiment_id"] == experiment_ids[2]]
    baseline_width_4_training_process_data = baseline_width_4_training_process_data[baseline_width_4_training_process_data["experiment_id"] == experiment_ids[2]]

    net2wider_width_4_callback_data_relevant = net2wider_smac_callback_data[net2wider_smac_callback_data["experiment_id"] == experiment_ids[3]]
    net2wider_width_4_training_process_data_relevant = net2wider_training_process_data[net2wider_training_process_data["experiment_id"] == experiment_ids[3]]

    net2wider_width_2_callback_data_relevant = net2wider_smac_callback_data[net2wider_smac_callback_data["experiment_id"] == experiment_ids[4]]
    net2wider_width_2_training_process_data_relevant = net2wider_training_process_data[net2wider_training_process_data["experiment_id"] == experiment_ids[4]]

    baseline_width1_smac_incumbent = baseline_width_1_callback_data.iloc[baseline_width_1_callback_data_relevant["cost"].idxmin()]
    baseline_width1_incumbent_trial_number = baseline_width1_smac_incumbent["trial_number"] - 1

    baseline_width2_smac_incumbent = baseline_width_2_callback_data.iloc[baseline_width_2_callback_data_relevant["cost"].idxmin()]
    baseline_width2_incumbent_trial_number = baseline_width2_smac_incumbent["trial_number"] - 1

    baseline_width4_smac_incumbent = baseline_width_4_callback_data.iloc[baseline_width_4_callback_data_relevant["cost"].idxmin()]
    baseline_width4_incumbent_trial_number = baseline_width4_smac_incumbent["trial_number"] - 1

    # Select all net2wider trialnumbers with the same hyperparameter string identifier as the incumbent
    net2wider_width_4_final_incumbent = net2wider_smac_callback_data.iloc[net2wider_width_4_callback_data_relevant["cost"].idxmin()]
    net2wider_width_4_hyperparameter_str_identifier = net2wider_width_4_final_incumbent["hyperparameter_str_identifier"]
    net2wider_width_4_incumbents = net2wider_width_4_callback_data_relevant[
        net2wider_width_4_callback_data_relevant["hyperparameter_str_identifier"] == net2wider_width_4_hyperparameter_str_identifier
    ]
    net2wider_width_4_incumbents_trial_numbers = net2wider_width_4_incumbents["trial_number"] - 1
    net2wider_width_4_incumbents_trial_numbers

    net2wider_width_2_final_incumbent = net2wider_smac_callback_data.iloc[net2wider_width_2_callback_data_relevant["cost"].idxmin()]
    net2wider_width_2_hyperparameter_str_identifier = net2wider_width_2_final_incumbent["hyperparameter_str_identifier"]
    net2wider_width_2_incumbents = net2wider_width_2_callback_data_relevant[
        net2wider_width_2_callback_data_relevant["hyperparameter_str_identifier"] == net2wider_width_2_hyperparameter_str_identifier
    ]
    net2wider_width_2_incumbents_trial_numbers = net2wider_width_2_incumbents["trial_number"] - 1
    net2wider_width_2_incumbents_trial_numbers

    baseline_width_1_training_process_data = baseline_width_1_training_process_data[baseline_width_1_training_process_data["trial_number"] == baseline_width1_incumbent_trial_number]
    baseline_width_2_training_process_data = baseline_width_2_training_process_data[baseline_width_2_training_process_data["trial_number"] == baseline_width2_incumbent_trial_number]
    net2wider_width_4_training_process_data = net2wider_width_4_training_process_data_relevant[
        net2wider_width_4_training_process_data_relevant["trial_number"].isin(net2wider_width_4_incumbents_trial_numbers)
    ]
    net2wider_width_2_training_process_data = net2wider_width_2_training_process_data_relevant[
        net2wider_width_2_training_process_data_relevant["trial_number"].isin(net2wider_width_2_incumbents_trial_numbers)
    ]

    max_timesteps_4 = net2wider_width_4_training_process_data["timestep"].max()
    max_timesteps_2 = net2wider_width_2_training_process_data["timestep"].max()

    net2wider_width_4_training_process_dataframes = []
    for i, trial_number in enumerate(net2wider_width_4_incumbents_trial_numbers):
        current = net2wider_width_4_training_process_data[net2wider_width_4_training_process_data["trial_number"] == trial_number]
        current["timestep"] += i * max_timesteps_4
        net2wider_width_4_training_process_dataframes.append(current)
    net2wider_width_4_training_process_concat_dataframe = pd.concat(net2wider_width_4_training_process_dataframes)

    net2wider_width_2_training_process_concat_dataframe = []
    for i, trial_number in enumerate(net2wider_width_2_incumbents_trial_numbers):
        current = net2wider_width_2_training_process_data[net2wider_width_2_training_process_data["trial_number"] == trial_number]
        current["timestep"] += i * max_timesteps_2
        net2wider_width_2_training_process_concat_dataframe.append(current)
    net2wider_width_2_training_process_concat_dataframe = pd.concat(net2wider_width_2_training_process_concat_dataframe)

    baseline_width_1_training_process_data = baseline_width_1_training_process_data[["timestep", "worker_id", "rewards_per_episode"]]
    baseline_width_2_training_process_data = baseline_width_2_training_process_data[["timestep", "worker_id", "rewards_per_episode"]]
    baseline_width_4_training_process_data = baseline_width_4_training_process_data[["timestep", "worker_id", "rewards_per_episode"]]
    net2wider_width_4_training_process_concat_dataframe = net2wider_width_4_training_process_concat_dataframe[["timestep", "worker_id", "rewards_per_episode"]]
    net2wider_width_2_training_process_concat_dataframe = net2wider_width_2_training_process_concat_dataframe[["timestep", "worker_id", "rewards_per_episode"]]

    baseline_width_1_training_process_data = convert_dataframe(baseline_width_1_training_process_data)
    baseline_width_2_training_process_data = convert_dataframe(baseline_width_2_training_process_data)
    baseline_width_4_training_process_data = convert_dataframe(baseline_width_4_training_process_data)
    net2wider_width_4_training_process_concat_dataframe = convert_dataframe(net2wider_width_4_training_process_concat_dataframe)
    net2wider_width_2_training_process_concat_dataframe = convert_dataframe(net2wider_width_2_training_process_concat_dataframe)

    return (
        baseline_width_1_training_process_data,
        baseline_width_2_training_process_data,
        baseline_width_4_training_process_data,
        net2wider_width_4_training_process_concat_dataframe,
        net2wider_width_2_training_process_concat_dataframe,
    )


def plot_10x10_random_full():
    (
        baseline_width_1_training_process_data,
        baseline_width_2_training_process_data,
        baseline_width_4_training_process_data,
        net2wider_width_4_training_process_concat_dataframe,
        net2wider_width_2_training_process_concat_dataframe,
    ) = get_data("fehring_growing_nn_new_seeded", [1, 2, 7, 8, 9])

    training_process_style()

    sns.lineplot(data=baseline_width_1_training_process_data, x="timestep", y="episode_reward", label="Constant Width 512", linestyle="-")
    sns.lineplot(data=baseline_width_2_training_process_data, x="timestep", y="episode_reward", label="Constant Width 1024", linestyle="--")
    sns.lineplot(data=baseline_width_4_training_process_data, x="timestep", y="episode_reward", label="Constant Width 4096", linestyle="-.")
    sns.lineplot(data=net2wider_width_4_training_process_concat_dataframe, x="timestep", y="episode_reward", label="Net2Wider Width 4096", linestyle=":")
    sns.lineplot(data=net2wider_width_2_training_process_concat_dataframe, x="timestep", y="episode_reward", label="Net2Wider Width 1024", linestyle=(0, (4, 1, 2, 1)))

    # Add vline every 500000 timesteps
    for i in range(1, int(net2wider_width_4_training_process_concat_dataframe["timestep"].max() / 500000)):
        plt.axvline(x=i * 500000, color="gray", linestyle="--")

    plt.title("Incumbent Training Process - Minihack Room Random 10x10", fontsize=18, fontweight="bold")
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("IQM of Evaluation Episode Reward", fontsize=14)

    plt.legend(title="Model Type", fontsize=12, title_fontsize=14, loc="center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2wider/incumbent_training_process/incumbent_training_process_random_10x10.png", bbox_inches="tight")


def plot_10x10_random_width2():
    (baseline_width_1_training_process_data, baseline_width_2_training_process_data, _, _, net2edeper_width_2_training_process_concat_dataframe) = get_data(
        "fehring_growing_nn_new_seeded", [1, 2, 7, 8, 9]
    )

    training_process_style()

    sns.lineplot(data=baseline_width_1_training_process_data, x="timestep", y="episode_reward", label="Constant Width 512", linestyle="-")
    sns.lineplot(data=baseline_width_2_training_process_data, x="timestep", y="episode_reward", label="Constant Width 1024", linestyle="--")
    sns.lineplot(data=net2edeper_width_2_training_process_concat_dataframe, x="timestep", y="episode_reward", label="Net2Wider Width 1024", linestyle=":")

    for i in range(1, int(net2edeper_width_2_training_process_concat_dataframe["timestep"].max() / 1000000)):
        plt.axvline(x=i * 1000000, color="gray", linestyle="--")

    plt.title("Width 2, Incumbent Training Process - Minihack Room Random 10x10", fontsize=18, fontweight="bold")
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("IQM of Evaluation Episode Reward", fontsize=14)

    plt.legend(title="Model Type", fontsize=12, title_fontsize=14, loc="center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2wider/incumbent_training_process/incumbent_training_process_random_10x10_width2.png", bbox_inches="tight")


def plot_10x10_random_width4():
    (baseline_width_1_training_process_data, _, baseline_width_4_training_process_data, net2wider_width_4_training_process_concat_dataframe, _) = get_data(
        "fehring_growing_nn_new_seeded", [1, 2, 7, 8, 9]
    )

    training_process_style()

    sns.lineplot(data=baseline_width_1_training_process_data, x="timestep", y="episode_reward", label="Constant Width 512", linestyle="-")
    sns.lineplot(data=baseline_width_4_training_process_data, x="timestep", y="episode_reward", label="Constant Width 4096", linestyle="-.")
    sns.lineplot(data=net2wider_width_4_training_process_concat_dataframe, x="timestep", y="episode_reward", label="Constant Width 4096", linestyle=":")

    for i in range(1, int(net2wider_width_4_training_process_concat_dataframe["timestep"].max() / 500000)):
        plt.axvline(x=i * 500000, color="gray", linestyle="--")

    plt.title("Width 4, Incumbent Training Process - Minihack Room Random 10x10", fontsize=18, fontweight="bold")
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("IQM of Evaluation Episode Reward", fontsize=14)

    plt.legend(title="Model Type", fontsize=12, title_fontsize=14, loc="center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2wider/incumbent_training_process/incumbent_training_process_random_10x10_width4.png", bbox_inches="tight")


def plot_10x10_monster_full():
    (
        baseline_width_1_training_process_data,
        baseline_width_2_training_process_data,
        baseline_width_4_training_process_data,
        net2wider_width_4_training_process_concat_dataframe,
        net2wider_width_2_training_process_concat_dataframe,
    ) = get_data("fehring_growing_nn_new_seeded", [3, 2, 8, 7, 10])

    training_process_style()

    sns.lineplot(data=baseline_width_1_training_process_data, x="timestep", y="episode_reward", label="Constant Width 512", linestyle="-")
    sns.lineplot(data=baseline_width_2_training_process_data, x="timestep", y="episode_reward", label="Constant Width 1024", linestyle="--")
    sns.lineplot(data=baseline_width_4_training_process_data, x="timestep", y="episode_reward", label="Constant Width 4096", linestyle="-.")
    sns.lineplot(data=net2wider_width_4_training_process_concat_dataframe, x="timestep", y="episode_reward", label="Net2Wider Width 4096", linestyle=":")
    sns.lineplot(data=net2wider_width_2_training_process_concat_dataframe, x="timestep", y="episode_reward", label="Net2Wider Width 1024", linestyle=(0, (4, 1, 2, 1)))

    # Add vline every 500000 timesteps
    for i in range(1, int(net2wider_width_4_training_process_concat_dataframe["timestep"].max() / 500000)):
        plt.axvline(x=i * 500000, color="gray", linestyle="--")

    plt.title("Incumbent Training Process - Minihack Room Monster 10x10", fontsize=18, fontweight="bold")
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("IQM of Evaluation Episode Reward", fontsize=14)

    plt.legend(title="Model Type", fontsize=12, title_fontsize=14, loc="center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2wider/incumbent_training_process/incumbent_training_process_monster_10x10.png", bbox_inches="tight")


def plot_10x10_monster_width2():
    (baseline_width_1_training_process_data, baseline_width_2_training_process_data, _, _, net2edeper_width_2_training_process_concat_dataframe) = get_data(
        "fehring_growing_nn_new_seeded", [3, 2, 8, 7, 10]
    )

    training_process_style()

    sns.lineplot(data=baseline_width_1_training_process_data, x="timestep", y="episode_reward", label="Constant Width 512", linestyle="-")
    sns.lineplot(data=baseline_width_2_training_process_data, x="timestep", y="episode_reward", label="Constant Width 1024", linestyle="--")
    sns.lineplot(data=net2edeper_width_2_training_process_concat_dataframe, x="timestep", y="episode_reward", label="Net2Wider Width 1024", linestyle=":")

    for i in range(1, int(net2edeper_width_2_training_process_concat_dataframe["timestep"].max() / 1000000)):
        plt.axvline(x=i * 1000000, color="gray", linestyle="--")

    plt.title("Width 2, Incumbent Training Process - Minihack Room Monster 10x10", fontsize=18, fontweight="bold")
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("IQM of Evaluation Episode Reward", fontsize=14)

    plt.legend(title="Model Type", fontsize=12, title_fontsize=14, loc="center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2wider/incumbent_training_process/incumbent_training_process_monster_10x10_width2.png", bbox_inches="tight")


def plot_10x10_monster_width4():
    (baseline_width_1_training_process_data, _, baseline_width_4_training_process_data, net2wider_width_4_training_process_concat_dataframe, _) = get_data(
        "fehring_growing_nn_new_seeded", [3, 2, 8, 7, 10]
    )

    training_process_style()

    sns.lineplot(data=baseline_width_1_training_process_data, x="timestep", y="episode_reward", label="Constant Width 512", linestyle="-")
    sns.lineplot(data=baseline_width_4_training_process_data, x="timestep", y="episode_reward", label="Constant Width 4096", linestyle="-.")
    sns.lineplot(data=net2wider_width_4_training_process_concat_dataframe, x="timestep", y="episode_reward", label="Net2Wider Width 4096", linestyle=":")

    for i in range(1, int(net2wider_width_4_training_process_concat_dataframe["timestep"].max() / 500000)):
        plt.axvline(x=i * 500000, color="gray", linestyle="--")

    plt.title("Width 4, Incumbent Training Process - Minihack Room Monster 10x10", fontsize=18, fontweight="bold")
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("IQM of Evaluation Episode Reward", fontsize=14)

    plt.legend(title="Model Type", fontsize=12, title_fontsize=14, loc="center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2wider/incumbent_training_process/incumbent_training_process_monster_10x10_width4.png", bbox_inches="tight")


plot_10x10_random_full()
plot_10x10_random_width2()
plot_10x10_random_width4()
plot_10x10_monster_full()
plot_10x10_monster_width2()
plot_10x10_monster_width4()
