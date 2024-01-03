"""
This module pre-processes the data from the filtered .pcap file and extracts the features that will be selected for the
predictions.
"""
from statistics import mean
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
from scapy.layers.dot11 import Dot11, RadioTap

PacketSet = List[Tuple[Any, Any, Dot11]]


def normalize(v: np.ndarray) -> np.ndarray:
    """
    This function returns the normalized input array.

    :param np.ndarray v: The input array
    :return: The normalized input array
    """
    norm = np.linalg.norm(v)
    return v if norm == 0 else v/norm


def cdf_value(v: np.ndarray, normalized: bool = True) -> float:
    """
    This function returns the CDF value over the input data series, as specified in the DeWiCam paper (section 5.3).
    :param bool normalized: If set to True, the CDF value will be computed over the normalized data series.
    :param np.ndarray v: The input array that contains the data series.
    :return:
    """
    data = np.sort(normalize(v)) if normalized else np.sort(v)
    sample = np.random.choice(a=data, size=round(len(data)/4), replace=False)
    # This is my interpretation of "over a small number of samples"
    return np.cumsum(sample)[-1]


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


def build_time_interval_array(timestamps: list) -> np.ndarray:
    """
    This function returns the Numpy Ndarray of the estimates of the transmission time of each packet.
    The estimate is computed as the difference between the timestamp of the current packet and timestamp of the
    previous one. The transmission time of the first packet is estimated as the mean of all the other values.

    :param list timestamps: List of the timestamps of the time of arrival of each packet.
    :return: Numpy Ndarray of the approximate transmission duration.
    """
    intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    return np.array([mean(intervals)] + intervals)


def feature_expansion(pkt_list: PacketSet) -> pd.DataFrame:
    """
    This function extract the main features from each packet's Wi-Fi header and collects everything into a Pandas
    DataFrame.
    :param pkt_list: List of the filtered packets from the .pcap file.
    :return: Pandas Dataframe of each packet and their main features.
    """
    pkt_length = list(map(lambda pkt, _, __: pkt, pkt_list))
    x = list(map(lambda _, __, pkt: dot11_to_feature(pkt), pkt_list))
    timestamps = list(map(lambda _, t, __: t, pkt_list))
    pd_x = pd.DataFrame(x, columns=["SourceAddress", "DestinationAddress", "Duration"])
    pd_x.loc[:, "PacketLength"] = pd.Series(pkt_length)
    pd_x.loc[:, "Timestamp"] = pd.Series(timestamps)
    pd_x.loc[:, "TimestampOffset"] = pd_x["Timestamp"] - pd_x["Timestamp"].min()
    # Timestamps are expressed as offsets from the first packet.
    pd_x.loc[:, "TransmissionTime"] = pd.Series(build_time_interval_array(timestamps))
    pd_x.loc[:, "Bandwidth"] = pd_x["PacketLength"] / pd_x["TransmissionTime"]
    return pd_x


def flow_creation(pkt_list: pd.DataFrame):
    """
    This function takes as input the Pandas DataFrame that contains the features of the captured packets, and returns
    a new DataFrame that groups the packets into flow, using a "group by" like operation. Then, the following features are
    extracted for each flow:
     * Cumulative Distribution Function of the normalized packet length.
     * Mean and s.d. of the “Duration” field.
     * Standard deviation of the flow bandwidth. (?)

    :param pkt_list: The Pandas DataFrame that contains the packets and their features.
    :return: A Pandas DataFrame that groups the input one into traffic flows and computes their aggregate features.
    """
    flow_pd = pd.DataFrame(index=pkt_list["SourceAddress"].unique(), columns=["CDF_npl", "Duration_mean", "Duration_sd", "Bandwidth_sd"])
    flow_pd.loc[:, "Duration_mean"] = pkt_list.groupby(by="SourceAddress")["Duration"].mean()
    flow_pd.loc[:, "Duration_sd"] = pkt_list.groupby(by="SourceAddress")["Duration"].sd()
    flow_pd.loc[:, "Bandwidth_sd"] = pkt_list.groupby(by="SourceAddress")["Bandwidth"].sd()
    flow_pd.loc[:, "CDF_npl"] = pkt_list.groupby(by="SourceAddress")["PacketLength"].apply(lambda x: cdf_value(x, False))
    return flow_pd
