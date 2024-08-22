import pandas as pd
import seaborn as sns
from utils.plotting import training_process_style, select_incumbents, get_logtable, set_rc_params
import matplotlib.pyplot as plt
from typing import List

set_rc_params()


def get_data(database_name: str, experiment_ids: List[int]) -> pd.DataFrame:
    baseline_width_1_callback_data = get_logtable(database_name=database_name, table_name="incumbent_gen_2_layers", logtable_name="smac_callbacks")
    baseline_width_1_callback_data = baseline_width_1_callback_data[baseline_width_1_callback_data["experiment_id"] == experiment_ids[0]]
    baseline_width_1_callback_data.head()

    baseline_width_2_callback_data = get_logtable(database_name=database_name, table_name="bb_net2wider_baseline", logtable_name="smac_callbacks")
    baseline_width_2_callback_data = baseline_width_2_callback_data[baseline_width_2_callback_data["experiment_id"] == experiment_ids[1]]
    baseline_width_2_callback_data.head()

    baseline_width_4_callback_data = get_logtable(database_name=database_name, table_name="bb_net2wider_baseline", logtable_name="smac_callbacks")
    baseline_width_4_callback_data = baseline_width_4_callback_data[baseline_width_4_callback_data["experiment_id"] == experiment_ids[2]]
    baseline_width_4_callback_data.head()

    net2wider_smac_callback_data = get_logtable(database_name=database_name, table_name="net2wider_budget200", logtable_name="smac_callbacks")
    net2wider_smac_callback_data_width_4 = net2wider_smac_callback_data[net2wider_smac_callback_data["experiment_id"] == experiment_ids[3]]
    net2wider_smac_callback_data_width_4.head()

    net2wider_smac_callback_data_width_2 = net2wider_smac_callback_data[net2wider_smac_callback_data["experiment_id"] == experiment_ids[4]]
    net2wider_smac_callback_data_width_2.head()

    baseline_width_1_callback_data = select_incumbents(baseline_width_1_callback_data)
    baseline_width_2_callback_data = select_incumbents(baseline_width_2_callback_data)
    baseline_width_4_callback_data = select_incumbents(baseline_width_4_callback_data)
    net2wider_smac_callback_data_width_2 = select_incumbents(net2wider_smac_callback_data_width_2)
    net2wider_smac_callback_data_width_4 = select_incumbents(net2wider_smac_callback_data_width_4)

    max_trial_number = [
        baseline_width_1_callback_data["trial_number"].max(),
        baseline_width_2_callback_data["trial_number"].max(),
        baseline_width_4_callback_data["trial_number"].max(),
        net2wider_smac_callback_data_width_2["trial_number"].max(),
        net2wider_smac_callback_data_width_4["trial_number"].max(),
    ]
    max_trial_number

    baseline_width_1_callback_data["environment_interactions"] = baseline_width_1_callback_data["trial_number"] * 2000000 + 2000000
    baseline_width_2_callback_data["environment_interactions"] = baseline_width_2_callback_data["trial_number"] * 2000000 + 2000000
    baseline_width_4_callback_data["environment_interactions"] = baseline_width_4_callback_data["trial_number"] * 2000000 + 2000000
    net2wider_smac_callback_data_width_4["environment_interactions"] = net2wider_smac_callback_data_width_4["trial_number"] * 500000 + 500000
    net2wider_smac_callback_data_width_2["environment_interactions"] = net2wider_smac_callback_data_width_2["trial_number"] * 1000000 + 1000000

    max_interactions = 50 * 2000000
    baseline_width_1_callback_data = pd.concat(
        [baseline_width_1_callback_data, pd.DataFrame(({"environment_interactions": max_interactions, "cost": baseline_width_1_callback_data["cost"].min(), "trial_number": 999999999},))],
        ignore_index=True,
    )
    baseline_width_2_callback_data = pd.concat(
        [baseline_width_2_callback_data, pd.DataFrame(({"environment_interactions": max_interactions, "cost": baseline_width_2_callback_data["cost"].min(), "trial_number": 999999999},))],
        ignore_index=True,
    )
    baseline_width_4_callback_data = pd.concat(
        [baseline_width_4_callback_data, pd.DataFrame(({"environment_interactions": max_interactions, "cost": baseline_width_4_callback_data["cost"].min(), "trial_number": 999999999},))],
        ignore_index=True,
    )
    net2wider_smac_callback_data_width_4 = pd.concat(
        [
            net2wider_smac_callback_data_width_4,
            pd.DataFrame(({"environment_interactions": max_interactions, "cost": net2wider_smac_callback_data_width_4["cost"].min(), "trial_number": 999999999},)),
        ],
        ignore_index=True,
    )
    net2wider_smac_callback_data_width_2 = pd.concat(
        [
            net2wider_smac_callback_data_width_2,
            pd.DataFrame(({"environment_interactions": max_interactions, "cost": net2wider_smac_callback_data_width_2["cost"].min(), "trial_number": 999999999},)),
        ],
        ignore_index=True,
    )

    return baseline_width_1_callback_data, baseline_width_2_callback_data, baseline_width_4_callback_data, net2wider_smac_callback_data_width_4, net2wider_smac_callback_data_width_2

def plot_10x10_random():
    baseline_width_1_callback_data, baseline_width_2_callback_data, baseline_width_4_callback_data, net2wider_smac_callback_data_width_4, net2wider_smac_callback_data_width_2 = get_data(
        "fehring_growing_nn_new_seeded", [1, 1, 4, 1, 4]
    )
    training_process_style()
    sns.lineplot(data=baseline_width_1_callback_data, x="environment_interactions", y="cost", label="Baseline (1 layer)", drawstyle="steps-post")
    sns.lineplot(data=baseline_width_2_callback_data, x="environment_interactions", y="cost", label="Baseline (2 layers)", drawstyle="steps-post")
    sns.lineplot(data=baseline_width_4_callback_data, x="environment_interactions", y="cost", label="Baseline (4 layers)", drawstyle="steps-post")
    sns.lineplot(data=net2wider_smac_callback_data_width_4, x="environment_interactions", y="cost", label="Net2Wider (4 layers)", drawstyle="steps-post")
    sns.lineplot(data=net2wider_smac_callback_data_width_2, x="environment_interactions", y="cost", label="Net2Wider (2 layers)", drawstyle="steps-post")

    plt.title("Optimization Process 10x10 Random", fontsize=18, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=14)
    plt.ylabel("Cost", fontsize=14)

    plt.legend(title="Model Type", fontsize=12, title_fontsize=14, loc="center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig("plotting/net2wider/overall_training_process/net2wider_training_process_random.png", bbox_inches="tight")

def plot_10x10_monster():
    baseline_width_1_callback_data, baseline_width_2_callback_data, baseline_width_4_callback_data, net2wider_smac_callback_data_width_4, net2wider_smac_callback_data_width_2 = get_data(
        "fehring_growing_nn_new_seeded", [8, 3, 6, 6, 5])
    
    training_process_style()
    sns.lineplot(data=baseline_width_1_callback_data, x="environment_interactions", y="cost", label="Baseline (1 layer)", drawstyle="steps-post")
    sns.lineplot(data=baseline_width_2_callback_data, x="environment_interactions", y="cost", label="Baseline (2 layers)", drawstyle="steps-post")
    sns.lineplot(data=baseline_width_4_callback_data, x="environment_interactions", y="cost", label="Baseline (4 layers)", drawstyle="steps-post")
    sns.lineplot(data=net2wider_smac_callback_data_width_4, x="environment_interactions", y="cost", label="Net2Wider (4 layers)", drawstyle="steps-post")
    sns.lineplot(data=net2wider_smac_callback_data_width_2, x="environment_interactions", y="cost", label="Net2Wider (2 layers)", drawstyle="steps-post")

    plt.title("Optimization Process 10x10 Monster", fontsize=18, fontweight="bold")
    plt.xlabel("Environment Interactions", fontsize=14)
    plt.ylabel("Cost", fontsize=14)

    plt.legend(title="Model Type", fontsize=12, title_fontsize=14, loc="center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)


    plt.savefig("plotting/net2wider/overall_training_process/net2wider_training_process_monster.png", bbox_inches="tight")

plot_10x10_random()
plot_10x10_monster()