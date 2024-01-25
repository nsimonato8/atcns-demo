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


def sample_audio(sound_capture: list[float], network_capture: pd.DataFrame, offset: int=0):
    """
    Downsamples the sound capture in such a way such the length of the list audio track and the list network capture are the same.

    Args:
        sound_capture (list[float]): List of floats that represents the sound samples.
        network_capture (pd.DataFrame): Pandas DataFrame of the cleaned captured file.
        offset (int, optional): The number of initial samples that you don't want to consider. Defaults to 0.

    Returns:
        list[float]: The downsampled audio file, as a list of floats.
    """
    kernel_size = int(len(sound_capture) / (network_capture.shape[0] - offset))
    sampled = []
    iters = network_capture.shape[0] - offset
    sampled += [0] * offset
    sampled += [sound_capture[i*kernel_size] for i in range(iters)]
    return sampled


def plot_audio_network_comparison(sound_capture: list[float], network_capture: pd.DataFrame):
    """Plots the comparison between the audio capture and the network capture.

    Args:
        sound_capture (list[float]): List of floats that represents the sound samples.
        network_capture (pd.DataFrame): Pandas DataFrame of the cleaned captured file.
    """
    sampled = sample_audio(sound_capture=sound_capture, capture=network_capture)
    plt.plot(list(range(len(sampled))), sampled, color='g', label="Pressure")
    plt.plot(list(range(len(sampled))), network_capture["PacketLength"], color='r', label="PacketLength")
    plt.title("Network traffic & Recorded sound comparison")
    plt.legend()
    pass
