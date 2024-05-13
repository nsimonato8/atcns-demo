"""
This file pre-processes the data directly from the capture files and organizes the data in both data flows and in packet lists.
"""
DATASET_NUMBER = 3
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_extraction import read_from_pcap, read_from_wav
from preprocessing import feature_expansion_raw
from preprocessing import flow_creation, binary_labeler

from utils import plot_packets_stat_distribution, plot_audio_network_comparison, sample_audio
from granger_causality import is_granger_caused, grangers_causation_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cleaner", description="This scripts reads, cleans and organizes the data into a dataset.")
    parser.add_argument("-S", "--source_address", type=str, required=True, help="set the MAC address to observe")
    parser.add_argument("-L", "--labels", type=str, required=True, help="set the labels of the datasets")
    parser.add_argument("-s", "--silent", type=bool, help="if set, does not print the figures")
    parser.add_argument("-a", "--audio", type=bool, help="if set, performs the causality studio on the audio tracks")
    
    args = parser.parse_args()
    
    # Reading the network data from the capture files
    captures_dir = "captures/" 
    captures_files = ["captures/network/" + f for f in os.listdir("captures/network")]
    captures = [read_from_pcap(path=f) for f in captures_files]
    
    # The network captures are pre-processed
    captures_pd = list(map(lambda x: feature_expansion_raw(x), captures))
    
    # Printing the distributions of the network captures
    plot_cols = ["Duration", "PacketLength", "TransmissionTime", "Bandwidth"]
    for c, f_c in zip(captures_pd, captures_files):
        plot_packets_stat_distribution(capture=c, filename=f_c, save_fig=not args.silent)
        
    # Cleaning the data and assigning the labels
    target_source_address = "0a:1a:cc:8e:ca:9e"
    labels = [(0,0), (0,0), (0,1), (0,0), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]
    flows_labeled_l = []

    for c, f_c, l in zip(captures_pd, captures_files, labels):
        filtered_capture = c.dropna(axis=0, inplace=False, subset=["SourceAddress"])
        filtered_capture.loc[filtered_capture["SourceAddress"] == target_source_address, :].to_csv(f"datasets/{f_c}_filtered.csv")
        flows = flow_creation(filtered_capture)
        flows_labeled = binary_labeler(flows=flows, criterion=target_source_address, label_name="IsMicrophone", label_values=l)
        flows_labeled.to_csv(f"datasets/{f_c}_flows_labeled.csv")
        flows_labeled_l.append(flows_labeled)
        
    # Concatenating the datasets
    for f in flows_labeled_l:
        f.loc[:, "SourceAddress"] = f.index

    full_dataset = pd.concat(flows_labeled_l, axis=0)
    full_dataset.to_csv(F"datasets/full_dataset_test-{DATASET_NUMBER}.csv")
    
    if args.audio:
        # Reading the audio data from the audio files
        audio_capture_files = []
        #ringtone_captures = [read_from_wav(filename=captures_dir + "audio/{f}.wav") for f in audio_capture_files] # This time the audio recording failed, so we'll use the ringtone with a trailing silence.
        ringtone_captures = [read_from_wav(filename=captures_dir + "audio/ringtone.wav")] * 5
        
        # Plots the comparsion between audio and network data
        offsets = [0.,0.,11.201,0.,4.987]
        for network, audio, offset in zip(captures_pd, ringtone_captures, offsets):
            plot_audio_network_comparison(sound_capture=audio[1], network_capture=network, offset=offset, sample_rate=audio[0])
        
        # Computes the Granger causality matrix
        grange_caus_mat = []
        for network, audio, offset in zip(captures_pd, ringtone_captures, offsets):
            sampled = sample_audio(sound_capture=audio[1], network_capture=network, offset=offset, sample_rate=audio[0])
            concat_pd = pd.concat([network.reset_index()["PacketLength"], pd.Series(sampled)], axis=1)
            concat_pd.columns = ["PacketLength", "Sound"]
            grange_caus_mat.append(grangers_causation_matrix(data=concat_pd, variables=concat_pd.columns, verbose=not args.silent, maxlag=5))
            
        with open(f"datasets/results/granger-causality-results_{DATASET_NUMBER}.txt", "w") as f:
            for matrix in grange_caus_mat:
                result = not is_granger_caused(granger_causality_table=matrix, y="PacketLength", x="Sound", threshold=.05)
                f.write(f"Result for Granger-Causality: {'Negative' if result else 'Positive'}\n\t Granger-Causality matrix:\n\t|{matrix[0,0]}|{matrix[0,1]}|\n\t|{matrix[1,0]}|{matrix[1,1]}|")
                f.write(f"\t{5 * '='}\n\n")
    