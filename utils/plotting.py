# make this a function
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def training_process_style():
    sns.color_palette("colorblind")
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
