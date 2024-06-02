"""
This module pre-processes the data from the filtered .pcap file and extracts the features that will be selected for the
predictions.
"""
from typing import List, Tuple, Any
import numpy as np
import os


import pandas as pd
from scapy.layers.dot11 import Dot11

PacketSet = List[Tuple[Any, Any, Dot11]]

features_cols = ["SourceAddress", "DestinationAddress", "PacketLength", "delta_PacketLength",
                 "TimestampOffset"] #, "Timestamp", "diff_Timestamp", "Duration", ]

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

def feature_expansion_raw(pkt_list: PacketSet, src: list[str]=["56:56:ee:a2:0f:a0", "3e:ba:76:31:02:ba"]) -> pd.DataFrame:
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
    
    # We filter just the packets from the observed device
    pd_x = pd_x.loc[pd_x["SourceAddress"].isin(src), :]
    
    pd_x.loc[:, "Timestamp"] = pd.Series(timestamps, dtype=float)    
    pd_x.loc[:, "TimestampSecond"] = pd_x["Timestamp"].apply(lambda x: round(x, 1))
    
    # Timestamps are expressed as offsets from the first packet.
    pd_x.loc[:, "TimestampOffset"] = 10. * (pd_x["Timestamp"] - pd_x["Timestamp"].min())
    
    pd_x.drop(labels=["Timestamp", "SourceAddress", "DestinationAddress"], axis=1, inplace=True)
    
    size_agg =  pd_x.groupby(['TimestampOffset']).size().reset_index(name="PacketRate")
    
    # size_agg now contains only the moments of non-zero packet rate, we fix that:
    pd_y = pd.DataFrame(np.arange(pd_x["TimestampOffset"].max()), dtype=int, columns=["TimestampOffset"])
    j = size_agg.join(other=pd_y, on="TimestampOffset", how="right", sort=True, lsuffix='_l')
    j.drop(labels=["TimestampOffset_l"], axis=1, inplace=True)

    return j.fillna(value=0., axis=1).set_index("TimestampOffset").to_numpy().tolist()