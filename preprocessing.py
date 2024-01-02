"""
This module pre-processes the data from the filtered .pcap file and extracts the features that will be selected for the
predictions.
"""
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
from scapy.layers.dot11 import Dot11

PacketSet = List[Tuple[Any, Dot11]]


def packet_to_entry(pkt: Dot11) -> np.ndarray:
    """
    This function returns a numpy ndarray that contains an encoding of the most important features of the 802.11 frame.
    The features are the following:
     * Source MAC Address (addr2).
     * Destination MAC Address (addr1).
     * Duration field (ID).

    :param Dot11 pkt: the packet to process.
    :return: The ndarray that encodes the input packet
    """
    return np.array([pkt.addr2, pkt.addr1, pkt.ID])


def feature_expansion(pkt_list: PacketSet) -> pd.DataFrame:
    """
    This function extract the main features from each packet's Wi-Fi header and collects everything into a Pandas
    DataFrame.
    :param pkt_list: List of the filtered packets from the .pcap file.
    :return: Pandas Dataframe of each packet and their main features.
    """
    x = list(map(lambda _, pkt: packet_to_entry(pkt), pkt_list))
    timestamps = list(map(lambda t, _: t, pkt_list))
    pd_x = pd.DataFrame(x, columns=["SourceAddress", "DestinationAddress", "Duration"])  # Initial feature setup.
    pd_x.loc[:, "Timestamp"] = pd.Series(timestamps)
    pd_x.loc[:, "Timestamp"] -= pd_x["Timestamp"].min()  # Timestamps are expressed as offsets from the first packet.
    return pd_x


def flow_creation(pkt_list: pd.DataFrame):
    # TODO: group by operation w.r.t. SourceAddress.
    # TODO: extract features from DeWiCam's paper.
    pass
