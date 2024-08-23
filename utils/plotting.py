# make this a function
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import io
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from py_experimenter.experimenter import PyExperimenter


def training_process_style():
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
    experimenter = PyExperimenter(config_file, database_name=database_name, table_name=table_name)
    return experimenter.get_table()


def get_logtable(database_name, table_name, logtable_name, config_file: str = "approach/minihack/net2deeper/config/net2deeper.yaml", use_ssh_tunnel=True):
    experimenter = PyExperimenter(config_file, database_name=database_name, table_name=table_name, use_ssh_tunnel=use_ssh_tunnel)
    return experimenter.get_logtable(logtable_name)


def set_rc_params():
    # Figure
    mpl.rcParams["figure.figsize"] = (6, 3)

    # Fontsizes
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["axes.titlesize"] = 12

    # Colors
    # - Seaborn Color Palette: colorblind
    # - default context always plotted in black

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")


def plotly_fig2array(fig):
    # convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png", scale=5)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def fig2img(fig, figsize=None, dpi=300):
    """Convert matplotlib figure to image as numpy array.

    :param fig: Plot to get image for.
    :type fig: matplotlib figure

    :param figsize: Optional figure size in inches, e.g. ``(10, 7)``.
    :type figsize: None or tuple of int

    :param dpi: Optional dpi.
    :type dpi: None or int

    :return: RGB image of plot
    :rtype: np.array
    """
    if dpi is not None:
        fig.set_dpi(dpi)
    if figsize is not None:
        fig.set_size_inches(figsize)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    image = np.reshape(image, (int(height), int(width), 3))

    # s, (width, height) = canvas.print_to_buffer()
    # image = np.fromstring(s, dtype=np.uint8).reshape((height, width, 3))

    return image


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
