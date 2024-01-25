"""
This file contains the functions that are used to plot the data and other secondary tasks.
"""
from matplotlib import pyplot as plt
import pandas as pd

def plot_packets_stat_distribution(captures: list[pd.DataFrame], filename: str, plot_cols: list[str]= ["Duration", "PacketLength", "TransmissionTime", "Bandwidth"]):
    """
    Plots the boxplots of each column of @captures.

    Args:
        captures (list[pd.DataFrame]): List of Pandas DataFrames of the cleaned capture files.
        filename (str): Name of the capture file.
        plot_cols (list[str], optional): List of the columns' names to plot. Defaults to ["Duration", "PacketLength", "TransmissionTime", "Bandwidth"].
    """
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(filename)
    for i, c in zip(axs.flat, plot_cols):
        captures[c].plot.box(ax=i)
    pass


def sample_audio(sound_capture: list[float], network_capture: pd.DataFrame):
    """
    Downsamples the sound capture in such a way such the length of the list audio track and the list network capture are the same.

    Args:
        sound_capture (list[float]): List of floats that represents the sound samples.
        network_capture (pd.DataFrame): Pandas DataFrame of the cleaned captured file.

    Returns:
        list[float]: The downsampled audio file, as a list of floats.
    """
    kernel_size = int(len(sound_capture) / network_capture.shape[0])
    return sound_capture[::kernel_size]


def plot_audio_network_comparison(sound_capture: list[float], network_capture: pd.DataFrame):
    """Plots the comparison between the audio capture and the network capture.

    Args:
        sound_capture (list[float]): List of floats that represents the sound samples.
        network_capture (pd.DataFrame): Pandas DataFrame of the cleaned captured file.
    """
    sampled = sample_audio(sound_capture=sound_capture, network_capture=network_capture)
    x_axis = list(range(len(sampled)))
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Time (s)')
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(x_axis, sampled, color=color, label="Pressure")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_axis[:network_capture.shape[0]], network_capture["PacketLength"], color=color, label="PacketLength")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Network traffic & Recorded sound comparison")
    plt.legend()
    plt.show()
    pass
