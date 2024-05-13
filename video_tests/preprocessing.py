"""
This module pre-processes the data from the filtered .pcap file and extracts the features that will be selected for the
predictions.
"""
from statistics import mean
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
from scapy.layers.dot11 import Dot11

PacketSet = List[Tuple[Any, Any, Dot11]]

def dot11_to_feature(pkt: Dot11) -> np.ndarray:
    """
    This function returns a numpy ndarray that contains an encoding of the most important features of the 802.11 frame.
    The features are the following:
     * Source MAC Address (addr2).
     * Destination MAC Address (addr1).
     * Duration field (ID).

    :param Dot11 pkt: the packet to process.
    :return: The ndarray that encodes the extracted features of the input packet.
    """
    return np.array([pkt.addr2, pkt.addr1, pkt.ID])

def feature_expansion_raw(pkt_list: PacketSet) -> pd.DataFrame:
    """
    This function extract the main features from each packet's Wi-Fi header and collects everything into a Pandas
    DataFrame.
    :param pkt_list: List of the filtered packets from the .pcap file.
    :return: Pandas Dataframe of each packet and their main features.
    """
    pkt_length = list(map(lambda pkt, *_,: pkt[0], pkt_list))
    x = list(map(lambda *pkt: dot11_to_feature(pkt[0][2]), pkt_list))
    timestamps = list(map(lambda *t: t[0][1], pkt_list))
    
    pd_x = pd.DataFrame(x, columns=["SourceAddress", "DestinationAddress", "Duration"])
    
    pd_x.loc[:, "Duration"] = pd_x.loc[:, "Duration"].astype(int)
    pd_x.loc[:, "PacketLength"] = pd.Series(pkt_length, dtype=int)
    pd_x.loc[:, "delta_PacketLength"] = np.diff(pd_x["PacketLength"].values, prepend=[0]) 
    
    # Timestamps are expressed as offsets from the first packet.
    pd_x.loc[:, "Timestamp"] = pd.Series(timestamps, dtype=float)
    pd_x.loc[:, "TimestampOffset"] = pd_x["Timestamp"] - pd_x["Timestamp"].min()
    pd_x.loc[:, "diff_Timestamp"] = np.diff(pd_x["Timestamp"].values, prepend=[0]) 
    
    return pd_x