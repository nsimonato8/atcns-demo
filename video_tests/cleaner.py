"""
This file pre-processes the data directly from the capture files and organizes the data in both data flows and in packet lists.
"""
DATASET_NUMBER = 2
import os
import gc
import pandas as pd
import argparse
from datetime import datetime
import numpy as np
from data_extraction import read_from_pcap
from preprocessing import feature_expansion_raw
from itertools import product 
from utils import plot_packets_stat_distribution, PLOT_COLS
import sys

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 100_000)
pd.set_option('display.max_columns', 100_000)

captures_dir = "video_tests/captures_video/" 
captures_kinds = ["lights", "no-lights", "daylight", "day-no-light"]

def main(services, silent):
    pd_datasets = pd.DataFrame([], columns=["X", "y", "service", "dark"])
    
    kind_services = product(services, captures_kinds)
    
    for service, kind  in kind_services:
        print(f"Processing [{service}][{kind}]")
        # Importing the captures
        # The network captures are pre-processed and the values of light and service are assigned
        print(f"\t[{kind}]")  
        path = captures_dir + service + "/" + f"{kind}/"      
        captures = map(lambda f: read_from_pcap(path=path + f), filter(lambda x: ".cap" in x, os.listdir(path)))
        captures = pd.DataFrame(map(lambda x: (feature_expansion_raw(x), int(not("day" in kind)), service, int("no" in kind)), captures), columns=["X", "y", "service", "dark"])
        pd_datasets = pd.concat([pd_datasets, captures], axis=0, ignore_index=True)
        del captures
        gc.collect()
        
    pd_datasets.to_csv(f"video_tests/datasets/services/dataset_{DATASET_NUMBER}_raw.csv", index=False)
        
    del pd_datasets
    gc.collect()
    print("Done!")            
    
    # Printing the distributions of the network captures
    if not silent:
        print("Printing the distributions of the data")
        for c, f_c in zip(pd_datasets, services):
            plot_packets_stat_distribution(capture=c, filename=f_c, save_fig=True)
        print("Done!")
    
    print("Done! Pre-processing completed")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cleaner", description="This scripts reads, cleans and organizes the data into a dataset.")
    parser.add_argument("-s", "--silent", action='store_true', help="if set, does not print the figures")
    
    args = parser.parse_args()    
    try:
        services = args.services
        services = services.split(",")
    except:
        services = ["Whatsapp", "Instagram", "Skype"]
        
    time_0 = datetime.now()
    print(main(services, args.silent))
    time_0 = time_0 - datetime.now()
    print(f"Execution time: {time_0} seconds")
    

    
