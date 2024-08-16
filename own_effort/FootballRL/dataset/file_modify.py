import pandas as pd
import numpy as np
import os
import glob

def calculate_channel_statistics(npy_files):
    num_channels=None
    min_vals=None
    max_vals=None

    for file in npy_files:
        data=np.load(file)
        if num_channels is None:
            num_channels=data.shape[2]
            min_vals=np.zeros(num_channels)
            max_vals=np.zeros(num_channels)

        for channel in range(num_channels):
            channel_data=data[:,:,channel]
            min_val=channel_data.min()
            max_val=channel_data.max()
            min_vals[channel]=min(min_val,min_vals[channel])
            max_vals[channel]=max(max_val,max_vals[channel])

    return min_vals,max_vals
def normalize_npy_file(data, min_vals, max_vals):
    num_channels=data.shape[2]
    for channel in range(num_channels):
        min_val=min_vals[channel]
        max_val=max_vals[channel]
        data[:,:,channel]=(data[:,:,channel]-min_val)/(max_val-min_val)
    return data

def normalize_npy_files(npy_files, output_dirs, min_vals, max_vals):
    os.makedirs(output_dirs,exist_ok=True)
    for file in npy_files:
        data=np.load(file)
        normalize_data=normalize_npy_file(data,min_vals,max_vals)
        output_path=os.path.join(output_dirs, os.path.basename(file))
        np.save(output_path,normalize_data)
if __name__=="__main__":
    npy_files_path="normalized_states/state_0.npy"
    state=np.load(npy_files_path)
    print(state.shape)
    for channel in range(state.shape[2]):
        print(f"Channel {channel} min: {state[:,:,channel].min()}, max: {state[:,:,channel].max()}")
