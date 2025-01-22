from utils.plotting import get_logtable, set_rc_params
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from utils.plotting import training_process_style, convert_dataframe
from typing import List

set_rc_params()


def get_data(database_name: str, experiment_ids: List[int]):
    database_name = "fehring_growing_nn_new_seeded"
    baseline_depth_1_callback_data = get_logtable(database_name=database_name, table_name="bb_net2deeper_baseline", logtable_name="smac_callbacks")
    baseline_depth_1_training_process_data = get_logtable(database_name=database_name, table_name="bb_net2deeper_baseline", logtable_name="training_process")

    baseline_depth_2_callback_data = get_logtable(database_name=database_name, table_name="incumbent_gen_2_layers", logtable_name="smac_callbacks")
    baseline_depth_2_training_process_data = get_logtable(database_name=database_name, table_name="incumbent_gen_2_layers", logtable_name="training_process")

    baseline_depth_4_callback_data = get_logtable(database_name=database_name, table_name="bb_net2deeper_baseline_4", logtable_name="smac_callbacks")
    baseline_depth_4_training_process_data = get_logtable(database_name=database_name, table_name="bb_net2deeper_baseline_4", logtable_name="training_process")

    net2deeper_smac_callback_data = get_logtable(database_name=database_name, table_name="net2deeper_budget200_final", logtable_name="smac_callbacks")
    net2deeper_training_process_data = get_logtable(database_name=database_name, table_name="net2deeper_budget200_final", logtable_name="training_process")

    baseline_depth_1_callback_data_relevant = baseline_depth_1_callback_data[baseline_depth_1_callback_data["experiment_id"] == experiment_ids[0]]
    baseline_depth_1_training_process_data = baseline_depth_1_training_process_data[baseline_depth_1_training_process_data["experiment_id"] == experiment_ids[0]]

    baseline_depth_2_callback_data_relevant = baseline_depth_2_callback_data[baseline_depth_2_callback_data["experiment_id"] == experiment_ids[1]]
    baseline_depth_2_training_process_data = baseline_depth_2_training_process_data[baseline_depth_2_training_process_data["experiment_id"] == experiment_ids[1]]

    baseline_depth_4_callback_data_relevant = baseline_depth_4_callback_data[baseline_depth_4_callback_data["experiment_id"] == experiment_ids[2]]
    baseline_depth_4_training_process_data = baseline_depth_4_training_process_data[baseline_depth_4_training_process_data["experiment_id"] == experiment_ids[2]]

    net2deeper_depth_4_callback_data_relevant = net2deeper_smac_callback_data[net2deeper_smac_callback_data["experiment_id"] == experiment_ids[3]]
    net2deeper_depth_4_training_process_data_relevant = net2deeper_training_process_data[net2deeper_training_process_data["experiment_id"] == experiment_ids[3]]

    net2deeper_depth_2_callback_data_relevant = net2deeper_smac_callback_data[net2deeper_smac_callback_data["experiment_id"] == experiment_ids[4]]
    net2deeper_depth_2_training_process_data_relevant = net2deeper_training_process_data[net2deeper_training_process_data["experiment_id"] == experiment_ids[4]]

    baseline_depth1_smac_incumbent = baseline_depth_1_callback_data.iloc[baseline_depth_1_callback_data_relevant["cost"].idxmin()]
    baseline_depth1_incumbent_trial_number = baseline_depth1_smac_incumbent["trial_number"] - 1

    baseline_depth2_smac_incumbent = baseline_depth_2_callback_data.iloc[baseline_depth_2_callback_data_relevant["cost"].idxmin()]
    baseline_depth2_incumbent_trial_number = baseline_depth2_smac_incumbent["trial_number"] - 1

    baseline_depth4_smac_incumbent = baseline_depth_4_callback_data.iloc[baseline_depth_4_callback_data_relevant["cost"].idxmin()]
    baseline_depth4_incumbent_trial_number = baseline_depth4_smac_incumbent["trial_number"] - 1

    # Select all net2deeper trialnumbers with the same hyperparameter string identifier as the incumbent
    net2deeper_depth_4_final_incumbent = net2deeper_smac_callback_data.iloc[net2deeper_depth_4_callback_data_relevant["cost"].idxmin()]
    net2deeper_depth_4_hyperparameter_str_identifier = net2deeper_depth_4_final_incumbent["hyperparameter_str_identifier"]
    net2deeper_depth_4_incumbents = net2deeper_depth_4_callback_data_relevant[
        net2deeper_depth_4_callback_data_relevant["hyperparameter_str_identifier"] == net2deeper_depth_4_hyperparameter_str_identifier
    ]
    net2deeper_depth_4_incumbents_trial_numbers = net2deeper_depth_4_incumbents["trial_number"] - 1
    net2deeper_depth_4_incumbents_trial_numbers

    net2deeper_depth_2_final_incumbent = net2deeper_smac_callback_data.iloc[net2deeper_depth_2_callback_data_relevant["cost"].idxmin()]
    net2deeper_depth_2_hyperparameter_str_identifier = net2deeper_depth_2_final_incumbent["hyperparameter_str_identifier"]
    net2deeper_depth_2_incumbents = net2deeper_depth_2_callback_data_relevant[
        net2deeper_depth_2_callback_data_relevant["hyperparameter_str_identifier"] == net2deeper_depth_2_hyperparameter_str_identifier
    ]
    net2deeper_depth_2_incumbents_trial_numbers = net2deeper_depth_2_incumbents["trial_number"] - 1
    net2deeper_depth_2_incumbents_trial_numbers

    baseline_depth_1_training_process_data = baseline_depth_1_training_process_data[baseline_depth_1_training_process_data["trial_number"] == baseline_depth1_incumbent_trial_number]
    baseline_depth_2_training_process_data = baseline_depth_2_training_process_data[baseline_depth_2_training_process_data["trial_number"] == baseline_depth2_incumbent_trial_number]
    net2deeper_depth_4_training_process_data = net2deeper_depth_4_training_process_data_relevant[
        net2deeper_depth_4_training_process_data_relevant["trial_number"].isin(net2deeper_depth_4_incumbents_trial_numbers)
    ]
    net2deeper_depth_2_training_process_data = net2deeper_depth_2_training_process_data_relevant[
        net2deeper_depth_2_training_process_data_relevant["trial_number"].isin(net2deeper_depth_2_incumbents_trial_numbers)
    ]

    max_timesteps_4 = net2deeper_depth_4_training_process_data["timestep"].max()
    max_timesteps_2 = net2deeper_depth_2_training_process_data["timestep"].max()

    net2deeper_depth_4_training_process_dataframes = []
    for i, trial_number in enumerate(net2deeper_depth_4_incumbents_trial_numbers):
        current = net2deeper_depth_4_training_process_data[net2deeper_depth_4_training_process_data["trial_number"] == trial_number]
        current["timestep"] += i * max_timesteps_4
        net2deeper_depth_4_training_process_dataframes.append(current)
    net2deeper_depth_4_training_process_concat_dataframe = pd.concat(net2deeper_depth_4_training_process_dataframes)

    net2deeper_depth_2_training_process_concat_dataframe = []
    for i, trial_number in enumerate(net2deeper_depth_2_incumbents_trial_numbers):
        current = net2deeper_depth_2_training_process_data[net2deeper_depth_2_training_process_data["trial_number"] == trial_number]
        current["timestep"] += i * max_timesteps_2
        net2deeper_depth_2_training_process_concat_dataframe.append(current)
    net2deeper_depth_2_training_process_concat_dataframe = pd.concat(net2deeper_depth_2_training_process_concat_dataframe)

    baseline_depth_1_training_process_data = baseline_depth_1_training_process_data[["timestep", "worker_id", "rewards_per_episode"]]
    baseline_depth_2_training_process_data = baseline_depth_2_training_process_data[["timestep", "worker_id", "rewards_per_episode"]]
    baseline_depth_4_training_process_data = baseline_depth_4_training_process_data[["timestep", "worker_id", "rewards_per_episode"]]
    net2deeper_depth_4_training_process_concat_dataframe = net2deeper_depth_4_training_process_concat_dataframe[["timestep", "worker_id", "rewards_per_episode"]]
    net2deeper_depth_2_training_process_concat_dataframe = net2deeper_depth_2_training_process_concat_dataframe[["timestep", "worker_id", "rewards_per_episode"]]

    baseline_depth_1_training_process_data = convert_dataframe(baseline_depth_1_training_process_data)
    baseline_depth_2_training_process_data = convert_dataframe(baseline_depth_2_training_process_data)
    baseline_depth_4_training_process_data = convert_dataframe(baseline_depth_4_training_process_data)
    net2deeper_depth_4_training_process_concat_dataframe = convert_dataframe(net2deeper_depth_4_training_process_concat_dataframe)
    net2deeper_depth_2_training_process_concat_dataframe = convert_dataframe(net2deeper_depth_2_training_process_concat_dataframe)

    return (
        baseline_depth_1_training_process_data,
        baseline_depth_2_training_process_data,
        baseline_depth_4_training_process_data,
        net2deeper_depth_4_training_process_concat_dataframe,
        net2deeper_depth_2_training_process_concat_dataframe,
    )


def plot_10x10_random_full():
    (
        baseline_depth_1_training_process_data,
        baseline_depth_2_training_process_data,
        baseline_depth_4_training_process_data,
        net2deeper_depth_4_training_process_concat_dataframe,
        net2deeper_depth_2_training_process_concat_dataframe,
    ) = get_data("fehring_growing_nn_new_seeded", [1, 1, 4, 1, 4])

    training_process_style()

    sns.lineplot(data=baseline_depth_2_training_process_data, x="timestep", y="episode_reward", label="Static (2 layers)", linestyle="solid")
    sns.lineplot(data=net2deeper_depth_2_training_process_concat_dataframe, x="timestep", y="episode_reward", label="GrowNN (2 layers)", linestyle="dotted", linewidth=3)
    sns.lineplot(data=baseline_depth_4_training_process_data, x="timestep", y="episode_reward", label="Static (4 layers)", linestyle="solid")
    sns.lineplot(data=net2deeper_depth_4_training_process_concat_dataframe, x="timestep", y="episode_reward", label="GrowNN (4 layers)", linestyle="dashed")
    sns.lineplot(data=baseline_depth_1_training_process_data, x="timestep", y="episode_reward", label="Static (1 layer) ", linestyle="solid")

    # Add vline every 500000 timesteps
    for i in range(1, int(net2deeper_depth_4_training_process_concat_dataframe["timestep"].max() / 500000)):
        plt.axvline(x=i * 500000, color="gray", linestyle="--")

    plt.xlabel("Environment Interactions", fontsize=25)
    plt.ylabel("IQM of Evaluation Episode Returns", fontsize=25)

    plt.legend(title="Model Type", fontsize=25, title_fontsize=25, loc="center", bbox_to_anchor=(0.45, -0.31), ncol=3)

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig("plotting/minihack/net2deeper/incumbent_training_process/incumbent_training_process_random_10x10.pdf", bbox_inches="tight", pad_inches=0.3)


def plot_10x10_random_depth2():
    (baseline_depth_1_training_process_data, baseline_depth_2_training_process_data, _, _, net2deeper_depth_2_training_process_concat_dataframe) = get_data(
        "fehring_growing_nn_new_seeded", [1, 1, 4, 1, 4]
    )

    training_process_style()

    sns.lineplot(data=baseline_depth_1_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 1", linestyle="solid")
    sns.lineplot(data=baseline_depth_2_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 2", linestyle="solid")
    sns.lineplot(data=net2deeper_depth_2_training_process_concat_dataframe, x="timestep", y="episode_reward", label="GrowNN Depth 2", linestyle="dotted", linewidth=3)

    for i in range(1, int(net2deeper_depth_2_training_process_concat_dataframe["timestep"].max() / 1000000)):
        plt.axvline(x=i * 1000000, color="gray", linestyle="--")

    plt.title("Depth 2, Incumbent Training Process - Minihack Room Random 10x10", fontsize=20, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=15)
    plt.ylabel("IQM of Evaluation Epsiode Return", fontsize=15)

    plt.legend(title="Model Type", fontsize=16, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2deeper/incumbent_training_process/incumbent_training_process_random_10x10_depth2.pdf", bbox_inches="tight", pad_inches=0.3)


def plot_10x10_random_depth4():
    (baseline_depth_1_training_process_data, _, baseline_depth_4_training_process_data, net2deeper_depth_4_training_process_concat_dataframe, _) = get_data(
        "fehring_growing_nn_new_seeded", [1, 1, 4, 1, 4]
    )

    training_process_style()

    sns.lineplot(data=baseline_depth_1_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 1", linestyle="solid")
    sns.lineplot(data=baseline_depth_4_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 4", linestyle="solid")
    sns.lineplot(data=net2deeper_depth_4_training_process_concat_dataframe, x="timestep", y="episode_reward", label="GrowNN Depth 4", linestyle="dashed")

    for i in range(1, int(net2deeper_depth_4_training_process_concat_dataframe["timestep"].max() / 500000)):
        plt.axvline(x=i * 500000, color="gray", linestyle="--")

    plt.title("Depth 4, Incumbent Training Process - Minihack Room Random 10x10", fontsize=20, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=15)
    plt.ylabel("IQM of Evaluation Epsiode Return", fontsize=15)

    plt.legend(title="Model Type", fontsize=16, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2deeper/incumbent_training_process/incumbent_training_process_random_10x10_depth4.png", bbox_inches="tight")


def plot_10x10_monster_full():
    (
        baseline_depth_1_training_process_data,
        baseline_depth_2_training_process_data,
        baseline_depth_4_training_process_data,
        net2deeper_depth_4_training_process_concat_dataframe,
        net2deeper_depth_2_training_process_concat_dataframe,
    ) = get_data("fehring_growing_nn_new_seeded", [8, 3, 6, 6, 5])

    training_process_style()

    sns.lineplot(data=baseline_depth_1_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 1", linestyle="solid")
    sns.lineplot(data=baseline_depth_2_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 2", linestyle="solid")
    sns.lineplot(data=net2deeper_depth_2_training_process_concat_dataframe, x="timestep", y="episode_reward", label="GrowNN Depth 2", linestyle="dotted", linewidth=3)
    sns.lineplot(data=baseline_depth_4_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 4", linestyle="solid")
    sns.lineplot(data=net2deeper_depth_4_training_process_concat_dataframe, x="timestep", y="episode_reward", label="GrowNN Depth 4", linestyle="dashed")

    # Add vline every 500000 timesteps
    for i in range(1, int(net2deeper_depth_4_training_process_concat_dataframe["timestep"].max() / 500000)):
        plt.axvline(x=i * 500000, color="gray", linestyle="--")

    plt.title("Incumbent Training Process - Minihack Room Monster 10x10", fontsize=20, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=15)
    plt.ylabel("IQM of Evaluation Epsiode Return", fontsize=15)

    plt.legend(title="Model Type", fontsize=16, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2deeper/incumbent_training_process/incumbent_training_process_monster_10x10.png", bbox_inches="tight")


def plot_10x10_monster_depth2():
    (baseline_depth_1_training_process_data, baseline_depth_2_training_process_data, _, _, net2deeper_depth_2_training_process_concat_dataframe) = get_data(
        "fehring_growing_nn_new_seeded", [8, 3, 6, 6, 5]
    )

    training_process_style()

    sns.lineplot(data=baseline_depth_1_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 1", linestyle="solid")
    sns.lineplot(data=baseline_depth_2_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 2", linestyle="solid")
    sns.lineplot(data=net2deeper_depth_2_training_process_concat_dataframe, x="timestep", y="episode_reward", label="GrowNN Depth 2", linestyle="dotted", linewidth=3)

    for i in range(1, int(net2deeper_depth_2_training_process_concat_dataframe["timestep"].max() / 1000000)):
        plt.axvline(x=i * 1000000, color="gray", linestyle="--")

    plt.title("Depth 2, Incumbent Training Process - Minihack Room Monster 10x10", fontsize=20, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=14)
    plt.ylabel("IQM of Evaluation Episode Returns", fontsize=14)

    plt.legend(title="Model Type", fontsize=12, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2deeper/incumbent_training_process/incumbent_training_process_monster_10x10_depth2.png", bbox_inches="tight")


def plot_10x10_monster_depth4():
    (baseline_depth_1_training_process_data, _, baseline_depth_4_training_process_data, net2deeper_depth_4_training_process_concat_dataframe, _) = get_data(
        "fehring_growing_nn_new_seeded", [8, 3, 6, 6, 5]
    )

    training_process_style()
    sns.lineplot(data=baseline_depth_1_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 1", linestyle="solid")
    sns.lineplot(data=baseline_depth_4_training_process_data, x="timestep", y="episode_reward", label="Constant Depth 4", linestyle="solid")
    sns.lineplot(data=net2deeper_depth_4_training_process_concat_dataframe, x="timestep", y="episode_reward", label="GrowNN Depth 4", linestyle="dashed")

    for i in range(1, int(net2deeper_depth_4_training_process_concat_dataframe["timestep"].max() / 500000)):
        plt.axvline(x=i * 500000, color="gray", linestyle="--")

    plt.title("Depth 4, Incumbent Training Process - Minihack Room Monster 10x10", fontsize=20, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=15)
    plt.ylabel("IQM of Evaluation Epsiode Return", fontsize=15)

    plt.legend(title="Model Type", fontsize=16, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()
    plt.savefig("plotting/minihack/net2deeper/incumbent_training_process/incumbent_training_process_monster_10x10_depth4.png", bbox_inches="tight")


plot_10x10_random_full()
# plot_10x10_random_depth2()
# plot_10x10_random_depth4()
# plot_10x10_monster_full()
# plot_10x10_monster_depth2()
# plot_10x10_monster_depth4()
