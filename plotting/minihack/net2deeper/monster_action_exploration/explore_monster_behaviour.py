# %%
import pandas as pd
import seaborn as sns
from utils.plotting import get_logtable, training_process_style, set_rc_params
from typing import List, Dict
from matplotlib import pyplot as plt


def calculate_n_dead_agents(rewards: Dict[int, List[float]]) -> int:
    n_dead = 0
    for vecenv_number, episodes in rewards.items():
        for episode in episodes:
            if len(episode) < 200:
                if episode[-1] == 0:
                    n_dead += 1 / 10
    return n_dead


set_rc_params()


def get_data(database_name, table_name, experiment_id: int):
    smac_data = get_logtable(database_name=database_name, table_name=table_name, logtable_name="smac_callbacks")
    training_process = get_logtable(database_name=database_name, table_name=table_name, logtable_name="training_process")
    monster_data: pd.DataFrame = smac_data[smac_data["experiment_id"] == experiment_id]

    # Select incumbent based on cost
    incumbent = monster_data[monster_data["cost"] == monster_data["cost"].min()]
    trial_number = incumbent["trial_number"].values[0] - 1

    monster_training_process = training_process[training_process["experiment_id"] == experiment_id]
    incumbent_training_process = monster_training_process[monster_training_process["trial_number"] == trial_number]

    incumbent_training_process = incumbent_training_process[["timestep", "worker_id", "rewards_per_episode"]]
    # sort by worker_id and timestep
    incumbent_training_process = incumbent_training_process.sort_values(by=["worker_id", "timestep"])

    dead_agents = dict()
    for worker_id in incumbent_training_process["worker_id"].unique():
        dead_agents[worker_id] = dict()
        worker_training_process = incumbent_training_process[incumbent_training_process["worker_id"] == worker_id]
        for row in worker_training_process.iterrows():
            rewards = row[1]["rewards_per_episode"]
            # Interpret str as dict
            rewards = eval(rewards)
            dead_agents[worker_id][row[1]["timestep"]] = calculate_n_dead_agents(rewards)

    dead_agents = pd.DataFrame(dead_agents)
    dead_agents = dead_agents.reset_index()
    return dead_agents


def plot_depth_4():
    database_name = "fehring_growing_nn_new_seeded"
    table_name = "bb_net2deeper_baseline_4"

    training_process_style()
    average_dead_agents = get_data(database_name, table_name, 6)
    for column in average_dead_agents.columns[1:]:
        sns.lineplot(x="index", y=column, data=average_dead_agents, label=f"Worker: {column}")

    plt.xlabel("Timesteps", fontsize=15)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.ylabel("% Dead Agents", fontsize=15)

    plt.title("Percentage Dead Agents, Net2Deeper Baselines Depth 4", fontsize=20, fontweight="bold")
    plt.legend(title="Baselines", fontsize=16, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("plotting/minihack/net2deeper/monster_action_exploration/dead_agents_depth_4.png", bbox_inches="tight")


def plot_depth_2():
    database_name = "fehring_growing_nn_new_seeded"
    table_name = "incumbent_gen_2_layers"

    training_process_style()
    average_dead_agents = get_data(database_name, table_name, 3)
    for column in average_dead_agents.columns[1:]:
        sns.lineplot(x="index", y=column, data=average_dead_agents, label=f"Worker: {column}")

    plt.xlabel("Timesteps", fontsize=15)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.ylabel("% Dead Agents", fontsize=15)

    plt.title("Percentage Dead Agents, Net2Deeper Baselines Depth 2", fontsize=20, fontweight="bold")
    plt.legend(title="Baselines", fontsize=16, title_fontsize=18, loc="center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("plotting/minihack/net2deeper/monster_action_exploration/dead_agents_depth_2.png", bbox_inches="tight")


plot_depth_2()
plot_depth_4()
