"""
This file pre-processes the data directly from the capture files and organizes the data in both data flows and in packet lists.
"""
DATASET_NUMBER = 1
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_extraction import read_from_pcap
from preprocessing import feature_expansion_raw

from utils import plot_packets_stat_distribution, PLOT_COLS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cleaner", description="This scripts reads, cleans and organizes the data into a dataset.")
    parser.add_argument("-S", "--source_address", type=str, required=True, help="set the MAC address to observe")
    parser.add_argument("-L", "--services", type=str, default="Whatsapp,Instagram,Skype", help="select the servies on which to perform the analysis")
    parser.add_argument("-s", "--silent", action='store_true', help="if set, does not print the figures")
    
    args = parser.parse_args()
    target_source_address = args.source_address
    # Preparing the network data from the capture files
    
    try:
        services = args.services
        services = services.split(",")
    except:
        services = ["Whatsapp", "Instagram", "Skype"]
    
    captures_dir = "captures_video/" 
    
    
    pd_datasets = []
    for service in services:
        print(f"Processing [{service}]")
        # Importing the captures
        capture_light = [read_from_pcap(path= captures_dir + service + "/" + "lights/" + f) for f in filter(lambda x: ".cap" in x, os.listdir(captures_dir + service + "/" + "lights/"))]
        captures_nolight = [read_from_pcap(path= captures_dir + service + "/" + "no-lights/" + f) for f in filter(lambda x: ".cap" in x, os.listdir(captures_dir + service + "/" + "no-lights/"))]
        
        # The network captures are pre-processed
        captures_light = list(map(lambda x: feature_expansion_raw(x), capture_light))
        captures_nolight = list(map(lambda x: feature_expansion_raw(x), captures_nolight))
        
        # Filtering the captures       
        captures_light = list(map(lambda x: x.loc[x["SourceAddress"] == target_source_address, :].to_numpy(), captures_light))
        captures_nolight = list(map(lambda x: x.loc[x["SourceAddress"] == target_source_address, :].to_numpy(), captures_nolight))
        
        # Assigning the values of light and service
        pd_captures_light = list(map(lambda x: (x, 1, service), captures_light))
        pd_captures_light = pd.DataFrame(pd_captures_light, columns=["X", "y", "service"])
        pd_captures_nolight = list(map(lambda x: (x, 0, service), captures_light))
        pd_captures_nolight = pd.DataFrame(pd_captures_nolight, columns=["X", "y", "service"])
        
        pd_datasets.append(pd.concat([pd_captures_light, pd_captures_nolight], axis=0).reset_index(drop=True))
        print("Done!")
        
        
    pd.concat(pd_datasets, axis=0).reset_index().to_csv(f"datasets/dataset_{DATASET_NUMBER}_raw.csv")
            
    
    # Printing the distributions of the network captures
    if not args.silent:
        print("Printing the distributions of the data")
        for c, f_c in zip(pd_datasets, services):
            plot_packets_stat_distribution(capture=c, filename=f_c, save_fig=True)
        print("Done!")
    
    print("Done! Pre-processing completed")