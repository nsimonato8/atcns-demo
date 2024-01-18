"""
This module does the following:
 - reads the .pcap file and filters the data of interest (filters 802.11 uplink Data packets).
 - reads the audio file and (possibly) performs some noise-reduction.
"""
from scapy.layers.dot11 import Dot11
from scapy.utils import PcapReader, tcpdump
from preprocessing import PacketSet

AP_MAC_ADDRESS = ""
BPF_FILTER = f"wlan type data subtype data && wlan.da == {AP_MAC_ADDRESS}"


def read_from_pcap(path: str, bpf_filter: str="") -> PacketSet:
    """
    This function reads the .pcap file and applies the BPF filter given in input.

    :param path: Path of the .pcap file to extract.
    :param bpf_filter: BPF filter to apply to the .pcap file.
    :return: List of filtered packets.
    """
    packet_stream = []
    with PcapReader(filename=path) as pcap_reader:
        for pkt in pcap_reader:
            dot11_packet = len(pkt), pkt.time, pkt
            packet_stream.append(dot11_packet)
    return packet_stream
