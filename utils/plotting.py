# make this a function
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import ast
import io
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from py_experimenter.experimenter import PyExperimenter


def training_process_style():
    """
    Set the style of the plot to be used for the training process.
    """
    palette = sns.color_palette("colorblind")
    sns.set_palette(palette)
    plt.tight_layout()
    plt.figure(figsize=(12, 8))

    def format_func(value, tick_number):
        if value >= 1e6:
            return f"{value / 1e6:.2f}M"
        elif value >= 1e3:
            return f"{value / 1e3:.2f}K"
        else:
            return str(int(value))

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))


def get_table(database_name, table_name, config_file: str = "approach/minihack/net2deeper/config/net2deeper.yaml"):
    """
    Get the table from the database with the given parameters.
    """
    experimenter = PyExperimenter(config_file, database_name=database_name, table_name=table_name)
    return experimenter.get_table()


def get_logtable(database_name, table_name, logtable_name, config_file: str = "approach/minihack/net2deeper/config/net2deeper.yaml", use_ssh_tunnel=True):
    """
    Get the logtable from the database with the given parameters.
    """
    experimenter = PyExperimenter(config_file, database_name=database_name, table_name=table_name, use_ssh_tunnel=use_ssh_tunnel)
    return experimenter.get_logtable(logtable_name)


def set_rc_params():
    """
    Set the rc parameters for the plots.
    """
    # Figure
    mpl.rcParams["figure.figsize"] = (6, 3)

    # Fontsizes
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["axes.titlesize"] = 14

    # Increase linewidth of plots
    mpl.rcParams["lines.linewidth"] = 2

    # Colors
    # - Seaborn Color Palette: colorblind
    # - default context always plotted in black

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")


def select_incumbents(smac_callbacks: pd.DataFrame) -> pd.DataFrame:
    "Given a dataframe contianing one training run, only keep the incumbent data points."
    relevant_columns = smac_callbacks[["trial_number", "cost"]]

    # Sort by trial_number to ensure proper order
    sorted_callbacks = relevant_columns.sort_values("trial_number")

    current_incumbent = float("inf")
    incumbent_data = []
    for index, row in sorted_callbacks.iterrows():
        if row["cost"] < current_incumbent:
            current_incumbent = row["cost"]
            incumbent_data.append(row)
    return pd.DataFrame(incumbent_data)


def convert_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns 'timestep', 'worker_id', and 'rewards_per_episode', convert it to a DataFrame with columns 'timestep', 'worker_id' and 'episode_reward',
    where the 'episode_reward' column contains the interquartile mean of the rewards_per_episode column.
    """

    def extract_keys(data_str):
        # Convert string to dictionary

        data_dict = ast.literal_eval(data_str)
        dat_dict_values = data_dict.values()
        new_values = []
        for venc_env_number in dat_dict_values:
            for evaluation_episode in venc_env_number:
                new_values.append(sum(evaluation_episode))
        # Return the dictionary (pandas will handle this correctly)
        new_values = {i: new_values[i] for i in range(len(new_values))}
        return pd.Series(new_values)

        # Apply the function to the column

    def interquartile_mean(group):
        """Generated using GPT 4o"""
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        filtered = group[(group >= q1) & (group <= q3)]
        return filtered.mean()

    df_extracted = dataframe["rewards_per_episode"].apply(extract_keys)

    # Combine the original DataFrame with the new one
    df_combined = pd.concat([dataframe, df_extracted], axis=1)
    df_combined = df_combined.set_index(["timestep", "worker_id"])

    df_combined.drop("rewards_per_episode", axis=1, inplace=True)

    df_combined = pd.DataFrame(df_combined.stack())
    df_combined.reset_index(inplace=True)
    del df_combined["level_2"]
    df_combined.columns = ["timestep", "worker_id", "episode_reward"]

    df_iqm = df_combined.groupby(["timestep", "worker_id"]).apply(interquartile_mean)
    df_iqm = df_iqm.reset_index(drop=True)
    df_iqm.columns = ["timestep", "worker_id", "episode_reward"]

    return df_iqm


def convert_dataframe_gym(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns 'timestep', 'worker_id', and 'all_costs', convert it to a DataFrame with columns 'timestep', 'worker_id' and 'episode_reward',
    where the 'episode_reward' column contains the interquartile mean of the rewards_per_episode column.
    """

    def extract_keys(data_str):
        # Convert string to dictionary

        data_dict = ast.literal_eval(",".join(data_str[1:-1].split()))

        return pd.Series(data_dict)

        # Apply the function to the column

    def interquartile_mean(group):
        """Generated using GPT 4o"""
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        filtered = group[(group >= q1) & (group <= q3)]
        return filtered.mean()

    df_extracted = dataframe["all_costs"].apply(extract_keys)

    # Combine the original DataFrame with the new one
    df_combined = pd.concat([dataframe, df_extracted], axis=1)
    df_combined = df_combined.set_index(["timestep", "worker_id"])

    df_combined.drop("all_costs", axis=1, inplace=True)

    df_combined = pd.DataFrame(df_combined.stack())
    df_combined.reset_index(inplace=True)
    del df_combined["level_2"]
    df_combined.columns = ["timestep", "worker_id", "episode_reward"]

    df_iqm = df_combined.groupby(["timestep", "worker_id"]).apply(interquartile_mean)
    df_iqm = df_iqm.reset_index(drop=True)
    df_iqm.columns = ["timestep", "worker_id", "episode_reward"]

    return df_iqm
