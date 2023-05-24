# Import necessary modules
import os
import mne
import pickle
from tqdm import tqdm
import argparse
import json
import time
import warnings


def read_eeg(file_path):
    # Load EEG data into an EEG raw object
    return mne.io.read_raw_brainvision(file_path, preload=True)


def set_channel_types(raw):
    # Set channel types for all channels to 'eeg'
    raw.set_channel_types({channel: 'eeg' for channel in raw.ch_names})
    # Correct channel types of LE and RE
    raw.set_channel_types({'LE':'misc', 'RE':'misc'})
    return raw


def get_filtered_eeg(file_path, filters):
    signal = read_eeg(file_path)
    signal = set_channel_types(signal)
    # Apply the specified filters
    for filter_type, freq in filters.items():
        if filter_type == 'lpf':
            # Apply a low-pass filter to the raw data
            signal = signal.filter(l_freq=None, h_freq=freq, fir_design='firwin')
        elif filter_type == 'hpf':
            # Apply a high-pass filter to the raw data
            signal = signal.filter(l_freq=freq, h_freq=None, fir_design='firwin')
        elif filter_type == 'nf':
            # Apply a notch filter to the raw data
            signal = signal.notch_filter(freqs=freq, notch_widths=0.5, method='spectrum_fit')
    return signal


def preprocess_data(input_data_path, output_data_path, filters):
    start_time = time.time()
    # Disable RuntimeWarning messages
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Split input and output paths into directories
    input_dirs = input_data_path.split('/' if '/' in input_data_path else '\\')
    output_dirs = output_data_path.split('/' if '/' in output_data_path else '\\')
    # Walk through the directory tree starting from input_data_path
    for dir_path, _, file_names in os.walk(input_data_path):
        # Split current directory path into individual directories
        dirs = dir_path.split('/' if '/' in dir_path else '\\')
        if 'preprocessed' in dirs:
            for file_name in tqdm(file_names):
                # Check if the file extension is '.vhdr'
                if os.path.splitext(file_name)[1] == ".vhdr":
                    # Get the full path of the current file
                    file_path = os.path.join(dir_path, file_name)
                    # Apply filtering to the EEG data using the specified filters
                    filtered_eeg = get_filtered_eeg(file_path, filters)
                    # Create the new directory path for the output file
                    new_dir_path = os.path.join(*output_dirs, *dirs[len(input_dirs):])
                    new_dir_path = os.path.abspath(new_dir_path)
                    os.makedirs(new_dir_path, exist_ok=True)
                    # Create the new file path for the output file with a '.pickle' extension
                    new_file_path = os.path.join(new_dir_path, os.path.splitext(file_name)[0] + '.pickle')
                    # Save the filtered EEG data as a pickle file
                    with open(new_file_path, mode='wb') as out_file:
                        pickle.dump(filtered_eeg, out_file, protocol=pickle.DEFAULT_PROTOCOL)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


def preprocess_from_command_line():
    # Creating an argument parser object
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    # Adding command line arguments
    parser.add_argument("input_data_path", help="Path to the input data directory")
    parser.add_argument("output_data_path", help="Path to the output data directory")
    parser.add_argument("--filters", default='{}', help="Optional key-value pairs in JSON format")
    # Parsing the command line arguments
    args = parser.parse_args()
    # Loading the JSON-formatted filters
    filters = json.loads(args.filters)
    # Preprocessing the data using the provided arguments
    preprocess_data(args.input_data_path, args.output_data_path, filters)


if __name__ == "__main__":
    preprocess_from_command_line()