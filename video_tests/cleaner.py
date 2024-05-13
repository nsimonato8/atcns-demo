"""
This file pre-processes the data directly from the capture files and organizes the data in both data flows and in packet lists.
"""
DATASET_NUMBER = 3
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_extraction import read_from_pcap
from preprocessing import feature_expansion_raw

from utils import plot_packets_stat_distribution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cleaner", description="This scripts reads, cleans and organizes the data into a dataset.")
    parser.add_argument("-S", "--source_address", type=str, required=True, help="set the MAC address to observe")
    parser.add_argument("-L", "--services", type=str, default="Whatsapp,Instagram,Skype", help="select the servies on which to perform the analysis")
    parser.add_argument("-s", "--silent", type=bool, help="if set, does not print the figures")
    
    args = parser.parse_args()
    
    # Preparing the network data from the capture files
    
    try:
        services = args.services
        services = services.split(",")
    except:
        services = ["Whatsapp", "Instagram", "Skype"]
    
    captures_dir = "captures_video/" 
    
    pd_datasets = []
    for service in services:
        # Importing the captures
        capture_light = [read_from_pcap(path= captures_dir + service + "/" + "light/" + f) for f in filter(lambda x: ".cap" in x, os.listdir(captures_dir + service + "/" + "light/"))]
        captures_nolight = [read_from_pcap(path= captures_dir + service + "/" + "no-lights/" + f) for f in filter(lambda x: ".cap" in x, os.listdir(captures_dir + service + "/" + "no-lights/"))]
        
        # The network captures are pre-processed
        captures_light = list(map(lambda x: feature_expansion_raw(x), capture_light))
        captures_nolight = list(map(lambda x: feature_expansion_raw(x), captures_nolight))
        
        # Assigning the values of light and service
        pd_captures_light = list(map(lambda x: x.assign(light=1, service=service), captures_light))
        pd_captures_light = pd.concat(pd_captures_light, axis=0)
        pd_captures_nolight = list(map(lambda x: x.assign(light=0, service=service), captures_light))
        pd_captures_nolight = pd.concat(pd_captures_nolight, axis=0)
        
        pd_datasets.append(pd.concat([pd_captures_light, pd_captures_nolight], axis=0))
        
        
    pd.concat(pd_datasets, axis=0).to_csv("datasets/dataset_{DATASET_NUMBER}_raw.csv")
            
    
    # Printing the distributions of the network captures
    if not args.silent:
        plot_cols = ["Duration", "PacketLength", "TransmissionTime", "delta_PacketLength"]
        for c, f_c in zip(pd_datasets, services):
            plot_packets_stat_distribution(capture=c, filename=f_c, save_fig=True)
    
        
    # Cleaning the data and assigning the labels
    target_source_address = args.source_address
    pd_datasets_filtered = []

    for c in pd_datasets:
        # Filtering the packets by filtering criteria
        filtered_capture = c.dropna(axis=0, inplace=False, subset=["SourceAddress"])
        
        # Filtering the packets by source address (we keep only the ones starting from out target device)
        filtered_capture = filtered_capture.loc[filtered_capture["SourceAddress"] == target_source_address, :]
        
        pd_datasets_filtered.append(filtered_capture)

    pd_datasets_filtered = pd.concat(pd_datasets_filtered, axis=0)
    pd_datasets_filtered.to_csv(F"datasets/dataset_{DATASET_NUMBER}_processed.csv")