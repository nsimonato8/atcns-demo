"""
This file contains the functions that are used to plot the data and other secondary tasks.
"""
from matplotlib import pyplot as plt
import pandas as pd

def plot_packets_stat_distribution(capture: list[pd.DataFrame], filename: str, plot_cols: list[str]= ["Duration", "PacketLength", "TransmissionTime", "Bandwidth"],
                                   save_fig: bool=False, save_dir: str="imgs"):
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
        capture[c].plot.box(ax=i)
        if save_fig:
            plt.savefig(f"{save_dir}/{filename}_packets_stat_distr.png")
    pass


def sample_audio(sound_capture: list[float], network_capture: pd.DataFrame, sample_rate: int=0, offset: float=0):
    """
    Downsamples the sound capture in such a way such the length of the list audio track and the list network capture are the same.
    To add a trailing silence to the track to sample, set the parameters sample_rate, offset to non-zero values.

    Args:
        sound_capture (list[float]): List of floats that represents the sound samples.
        network_capture (pd.DataFrame): Pandas DataFrame of the cleaned captured file.
        sample_rate (int): number of audio samples collected per seconds.
        offset (float): duration of the trailing silence in seconds.
        
    Returns:
        list[float]: The downsampled audio file, as a list of floats.
    """
    res = [0.] * int(sample_rate * offset)
    res.extend(sound_capture)
    kernel_size = int(len(res) / network_capture.shape[0])
    return res[::kernel_size]


def plot_audio_network_comparison(sound_capture: list[float], network_capture: pd.DataFrame, sample_rate: int=0, offset: float=0, 
                                  save_fig: bool=False, save_dir: str="imgs"):
    """
    Plots the comparison between the audio capture and the network capture.

    Args:
        sound_capture (list[float]): List of floats that represents the sound samples.
        network_capture (pd.DataFrame): Pandas DataFrame of the cleaned captured file.
        sample_rate (int): number of audio samples collected per seconds.
        offset (float): duration of the trailing silence in seconds.
    """
    sampled = sample_audio(sound_capture=sound_capture, network_capture=network_capture, sample_rate=sample_rate, offset=offset)
    x_axis = list(range(len(sampled)))
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Time (s)')
    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude', color=color)
    ax1.plot(x_axis, sampled, color=color, label="Pressure")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Packet length (bytes)', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_axis[:network_capture.shape[0]], network_capture["PacketLength"], color=color, alpha=0.5, label="PacketLength")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([-network_capture["PacketLength"].max(), network_capture["PacketLength"].max()])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Network traffic & Recorded sound comparison")
    plt.legend()
    
    if save_fig:
        plt.savefig(f"{save_dir}/audio_network_comparison.png")
    
    plt.show()
    pass
