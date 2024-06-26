"""
This module does the following:
 - reads the .pcap file and filters the data of interest (filters 802.11 uplink Data packets).
 - reads the audio file and (possibly) performs some noise-reduction.
"""
from typing import Tuple
import numpy as np
from scipy.io.wavfile import read as read_audio_file
from scapy.layers.dot11 import Dot11
from scapy.utils import PcapReader

from preprocessing import PacketSet


AudioRecording = Tuple[int, np.ndarray]

AP_MAC_ADDRESS = ""
BPF_FILTER = f"wlan.sa == {AP_MAC_ADDRESS}"


def read_from_pcap(path: str) -> PacketSet:
    """
    This function reads the .pcap file and applies the BPF filter given in input.

    :param path: Path of the .pcap file to extract.
    :return: List of filtered packets.
    """
    packet_stream = []
    with PcapReader(filename=path) as pcap_reader:
        for pkt in pcap_reader:
            dot11_packet = len(pkt), pkt.time, pkt
            packet_stream.append(dot11_packet)
    return packet_stream


def read_from_wav(filename: str) -> AudioRecording:
    """
    This function reads the audio data from a .wav file and returns it as an AudioRecording.
    Throws an AssertionError if you try to read a non-.wav file.

    :param filename: The path of the .wav audio file to read.
    :return: AudioRecording encoding of the audio file, that is a (bit rate, values) tuple of type (int, Numpy Ndarray).
    """
    assert len(filename) > 4, f"'{filename}' is not a valid .wav file path."
    assert filename[-4:] == ".wav", "'filename' should be the path to a .wav file."
    return read_audio_file(filename=filename)
