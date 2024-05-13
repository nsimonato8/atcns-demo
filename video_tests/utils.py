"""
This file contains the functions that are used to plot the data and other secondary tasks.
"""
from matplotlib import pyplot as plt
import pandas as pd

PLOT_COLS = ["Duration", "PacketLength", "TransmissionTime", "delta_PacketLength"]

def plot_packets_stat_distribution(capture: list[pd.DataFrame], filename: str, plot_cols: list[str]=PLOT_COLS,
                                   save_fig: bool=False, save_dir: str="imgs"):
    """
    Plots the boxplots of each column of @captures.

    Args:
        captures (list[pd.DataFrame]): List of Pandas DataFrames of the cleaned capture files.
        filename (str): Name of the capture file.
        plot_cols (list[str], optional): List of the columns' names to plot. Defaults to ["Duration", "PacketLength", "TransmissionTime", "delta_PacketLength"].
    """
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(filename)
    for i, c in zip(axs.flat, plot_cols):
        capture[c].plot.box(ax=i)
        if save_fig:
            plt.savefig(f"{save_dir}/{filename}_packets_stat_distr.png")
    pass