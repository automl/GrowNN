import pandas as pd
import seaborn as sns
from utils.plotting import training_process_style, select_incumbents, get_logtable, set_rc_params
import matplotlib.pyplot as plt
from typing import List

set_rc_params()


def get_data(database_name: str, experiment_ids: List[int]) -> pd.DataFrame:
    baseline_depth_1_callback_data = get_logtable(database_name=database_name, table_name="bb_net2deeper_baseline", logtable_name="smac_callbacks")
    baseline_depth_1_callback_data = baseline_depth_1_callback_data[baseline_depth_1_callback_data["experiment_id"] == experiment_ids[0]]
    baseline_depth_1_callback_data.head()

    baseline_depth_2_callback_data = get_logtable(database_name=database_name, table_name="incumbent_gen_2_layers", logtable_name="smac_callbacks")
    baseline_depth_2_callback_data = baseline_depth_2_callback_data[baseline_depth_2_callback_data["experiment_id"] == experiment_ids[1]]
    baseline_depth_2_callback_data.head()

    baseline_depth_4_callback_data = get_logtable(database_name=database_name, table_name="bb_net2deeper_baseline_4", logtable_name="smac_callbacks")
    baseline_depth_4_callback_data = baseline_depth_4_callback_data[baseline_depth_4_callback_data["experiment_id"] == experiment_ids[2]]
    baseline_depth_4_callback_data.head()

    net2deeper_smac_callback_data = get_logtable(database_name=database_name, table_name="net2deeper_budget200_final", logtable_name="smac_callbacks")
    net2deeper_smac_callback_data_depth_4 = net2deeper_smac_callback_data[net2deeper_smac_callback_data["experiment_id"] == experiment_ids[3]]
    net2deeper_smac_callback_data_depth_4.head()

    net2deeper_smac_callback_data_depth_2 = net2deeper_smac_callback_data[net2deeper_smac_callback_data["experiment_id"] == experiment_ids[4]]
    net2deeper_smac_callback_data_depth_2.head()

    baseline_depth_1_callback_data = select_incumbents(baseline_depth_1_callback_data)
    baseline_depth_2_callback_data = select_incumbents(baseline_depth_2_callback_data)
    baseline_depth_4_callback_data = select_incumbents(baseline_depth_4_callback_data)
    net2deeper_smac_callback_data_depth_2 = select_incumbents(net2deeper_smac_callback_data_depth_2)
    net2deeper_smac_callback_data_depth_4 = select_incumbents(net2deeper_smac_callback_data_depth_4)

    max_trial_number = [
        baseline_depth_1_callback_data["trial_number"].max(),
        baseline_depth_2_callback_data["trial_number"].max(),
        baseline_depth_4_callback_data["trial_number"].max(),
        net2deeper_smac_callback_data_depth_2["trial_number"].max(),
        net2deeper_smac_callback_data_depth_4["trial_number"].max(),
    ]
    max_trial_number

    baseline_depth_1_callback_data["environment_interactions"] = baseline_depth_1_callback_data["trial_number"] * 2000000
    baseline_depth_2_callback_data["environment_interactions"] = baseline_depth_2_callback_data["trial_number"] * 2000000
    baseline_depth_4_callback_data["environment_interactions"] = baseline_depth_4_callback_data["trial_number"] * 2000000
    net2deeper_smac_callback_data_depth_4["environment_interactions"] = net2deeper_smac_callback_data_depth_4["trial_number"] * 500000
    net2deeper_smac_callback_data_depth_2["environment_interactions"] = net2deeper_smac_callback_data_depth_2["trial_number"] * 1000000

    max_interactions = 50 * 2000000
    baseline_depth_1_callback_data = pd.concat(
        [baseline_depth_1_callback_data, pd.DataFrame(({"environment_interactions": max_interactions, "cost": baseline_depth_1_callback_data["cost"].min(), "trial_number": 999999999},))],
        ignore_index=True,
    )
    baseline_depth_2_callback_data = pd.concat(
        [baseline_depth_2_callback_data, pd.DataFrame(({"environment_interactions": max_interactions, "cost": baseline_depth_2_callback_data["cost"].min(), "trial_number": 999999999},))],
        ignore_index=True,
    )
    baseline_depth_4_callback_data = pd.concat(
        [baseline_depth_4_callback_data, pd.DataFrame(({"environment_interactions": max_interactions, "cost": baseline_depth_4_callback_data["cost"].min(), "trial_number": 999999999},))],
        ignore_index=True,
    )
    net2deeper_smac_callback_data_depth_4 = pd.concat(
        [
            net2deeper_smac_callback_data_depth_4,
            pd.DataFrame(({"environment_interactions": max_interactions, "cost": net2deeper_smac_callback_data_depth_4["cost"].min(), "trial_number": 999999999},)),
        ],
        ignore_index=True,
    )
    net2deeper_smac_callback_data_depth_2 = pd.concat(
        [
            net2deeper_smac_callback_data_depth_2,
            pd.DataFrame(({"environment_interactions": max_interactions, "cost": net2deeper_smac_callback_data_depth_2["cost"].min(), "trial_number": 999999999},)),
        ],
        ignore_index=True,
    )

    return baseline_depth_1_callback_data, baseline_depth_2_callback_data, baseline_depth_4_callback_data, net2deeper_smac_callback_data_depth_4, net2deeper_smac_callback_data_depth_2


def plot_10x10_random():
    baseline_depth_1_callback_data, baseline_depth_2_callback_data, baseline_depth_4_callback_data, net2deeper_smac_callback_data_depth_4, net2deeper_smac_callback_data_depth_2 = get_data(
        "fehring_growing_nn_new_seeded", [1, 1, 4, 1, 4]
    )
    training_process_style()
    sns.lineplot(
        data=baseline_depth_1_callback_data,
        x="environment_interactions",
        y="cost",
        label="Baseline (1 layer)",
        drawstyle="steps-post",
        linestyle="solid",
    )
    sns.lineplot(data=baseline_depth_2_callback_data, x="environment_interactions", y="cost", label="Baseline (2 layers)", drawstyle="steps-post", linestyle="solid")
    sns.lineplot(data=net2deeper_smac_callback_data_depth_2, x="environment_interactions", y="cost", label="Net2Deeper (2 layers)", drawstyle="steps-post", linestyle="dotted", linewidth=3)
    sns.lineplot(data=baseline_depth_4_callback_data, x="environment_interactions", y="cost", label="Baseline (4 layers)", drawstyle="steps-post", linestyle="solid")
    sns.lineplot(data=net2deeper_smac_callback_data_depth_4, x="environment_interactions", y="cost", label="Net2Deeper (4 layers)", drawstyle="steps-post", linestyle="dashed")

    plt.title("Optimization Process 10x10 Random", fontsize=20, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=15)
    plt.ylabel("Cost; Negative Mean Evaluation Return", fontsize=15)

    plt.legend(title="Model Type", fontsize=16, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig("plotting/minihack/net2deeper/overall_training_process/net2deeper_training_process_random.png", bbox_inches="tight")


def plot_10x10_monster():
    baseline_depth_1_callback_data, baseline_depth_2_callback_data, baseline_depth_4_callback_data, net2deeper_smac_callback_data_depth_4, net2deeper_smac_callback_data_depth_2 = get_data(
        "fehring_growing_nn_new_seeded", [8, 3, 6, 6, 5]
    )

    training_process_style()
    sns.lineplot(
        data=baseline_depth_1_callback_data,
        x="environment_interactions",
        y="cost",
        label="Baseline (1 layer)",
        drawstyle="steps-post",
        linestyle="solid",
    )
    sns.lineplot(data=baseline_depth_2_callback_data, x="environment_interactions", y="cost", label="Baseline (2 layers)", drawstyle="steps-post", linestyle="solid")
    sns.lineplot(data=net2deeper_smac_callback_data_depth_2, x="environment_interactions", y="cost", label="Net2Deeper (2 layers)", drawstyle="steps-post", linestyle="dotted", linewidth=3)
    sns.lineplot(data=baseline_depth_4_callback_data, x="environment_interactions", y="cost", label="Baseline (4 layers)", drawstyle="steps-post", linestyle="solid")
    sns.lineplot(data=net2deeper_smac_callback_data_depth_4, x="environment_interactions", y="cost", label="Net2Deeper (4 layers)", drawstyle="steps-post", linestyle="dashed")

    plt.title("Optimization Process 10x10 Monster", fontsize=20, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=15)
    plt.ylabel("Cost; Negative Mean Evaluation Return", fontsize=15)

    plt.legend(title="Model Type", fontsize=16, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig("plotting/minihack/net2deeper/overall_training_process/net2deeper_training_process_monster.png", bbox_inches="tight")


plot_10x10_random()
plot_10x10_monster()
