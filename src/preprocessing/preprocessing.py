# Import necessary modules
import os
import mne
import pickle
from tqdm import tqdm
import argparse
import json
import time
import warnings
import multiprocessing

import sys
sys.path.append('../utils/')
from format_hbn import HBNFormatter


def get_hbn_formatter(datasets_path):
    montage_bap = mne.channels.make_standard_montage("standard_1020")
    custom_montage_path = os.path.join(datasets_path, 'hbn', 'raw', 'GSN_HydroCel_129.tsv')
    montage_hbn = mne.channels.read_custom_montage(custom_montage_path)
    channel_names_bap = ['Fp1',  'Fp2',  'F3',  'F4',  'C3',  'C4',  'P3',  'P4',  'O1',  'O2',  
                         'F7',  'F8',  'T7',  'T8',  'P7',  'P8',  'Fz',  'Cz',  'Pz',  'Oz',  
                         'FC1',  'FC2',  'CP1',  'CP2',  'FC5',  'FC6',  'CP5',  'CP6',  'TP9',  
                         'TP10',  'P1',  'P2',  'C1',  'C2',  'FT9',  'FT10',  'AF3',  'AF4',  
                         'FC3',  'FC4',  'CP3',  'CP4',  'PO3',  'PO4',  'F5',  'F6',  'C5',  'C6',
                         'P5',  'P6',  'PO9',  'Iz',  'FT7',  'FT8',  'TP7',  'TP8',  'PO7', 
                         'PO8',  'Fpz',  'PO10',  'CPz',  'POz',  'FCz']
    
    return HBNFormatter(channel_names_bap=channel_names_bap, montage_bap=montage_bap, montage_hbn=montage_hbn)


def preprocess_file(file_path, dataset_name, hbn_formatter, new_file_path=None, sfreq=None, references=[], filters={}):    
    eeg_obj = None
    if dataset_name == 'bap':
        # Load EEG data into an EEG raw object
        eeg_obj = mne.io.read_raw_brainvision(file_path, verbose=False, preload=True)
        # Set channel types for all channels to 'eeg'
        eeg_obj.set_channel_types({channel: 'eeg' for channel in eeg_obj.ch_names})
        # Correct channel types of LE and RE
        eeg_obj.set_channel_types({'LE':'misc', 'RE':'misc'})
    elif dataset_name == 'hbn':
        eeg_obj = hbn_formatter(file_path)

    for ref_item in references:
        if ref_item == 'average':
            channel_names = eeg_obj.ch_names
            eeg_obj.set_eeg_reference(ref_channels="average")
            eeg_obj = eeg_obj.pick(picks=channel_names)
        # elif ref_item == 'median':
        #     median_channel = np.median(eeg_obj.get_data(), axis=0)
        #     eeg_obj -= median_channel
        #     pass # TODO
        elif ref_item in channel_names:
            channel_names = eeg_obj.ch_names
            eeg_obj.set_eeg_reference(ref_channels=ref_item)
            eeg_obj = eeg_obj.pick(picks=channel_names)

    # Apply the specified filters
    # Note: The if-else is implemented this way to minimize the number filtering operations
    if 'nf' in filters:
        # Apply a notch filter to the raw data
        eeg_obj = eeg_obj.notch_filter(freqs=filters['nf'], notch_widths=0.5, method='spectrum_fit')
    if 'lpf' in filters and 'hpf' in filters:
        # Apply a band-pass filter to the raw data
        eeg_obj = eeg_obj.filter(l_freq=filters['hpf'], h_freq=filters['lpf'], fir_design='firwin')
    elif 'lpf' in filters:
        # Apply a low-pass filter to the raw data
        eeg_obj = eeg_obj.filter(l_freq=None, h_freq=filters['lpf'], fir_design='firwin')
    elif 'hpf' in filters:
        # Apply a high-pass filter to the raw data
        eeg_obj = eeg_obj.filter(l_freq=filters['hpf'], h_freq=None, fir_design='firwin')

    if sfreq is not None:
        eeg_obj = eeg_obj.resample(sfreq=sfreq)

    if new_file_path is not None:
        # Save the filtered EEG data as a pickle file
        with open(new_file_path, mode='wb') as out_file:
            pickle.dump(eeg_obj, out_file, protocol=pickle.DEFAULT_PROTOCOL)
        
    return eeg_obj.get_data(), None if new_file_path is None else None, new_file_path


def callback(new_file_path):
    print(f"Saved: {new_file_path}")


def preprocessed_data(datasets_path, dataset_names, sfreq, references, filters):
    start_time = time.time()

    # Disable RuntimeWarning messages
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Split datasets_path into directories
    datasets_path_dirs = datasets_path.split('/' if '/' in datasets_path else '\\')

    hbn_formatter = get_hbn_formatter(datasets_path)

    # Set the number of worker processes (adjust according to your system)
    num_processes = 32
    # num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        for dataset_name in dataset_names:
            input_data_dirs = None
            if dataset_name == 'hbn':
                input_data_dirs = datasets_path_dirs + ['hbn', 'unzipped']
            elif dataset_name == 'bap':
                input_data_dirs = datasets_path_dirs + ['bap', 'raw']
            
            input_data_path = os.path.join('/', *input_data_dirs)

            output_data_dirs = data_ = datasets_path_dirs + [dataset_name, 'preprocessed']
            
            # Walk through the directory tree starting from input_data_path
            for dir_path, _, file_names in os.walk(input_data_path):
                
                # Split current directory path into individual directories
                cur_dirs = dir_path.split('/' if '/' in dir_path else '\\')
                
                if dataset_name == 'hbn' or (dataset_name == 'bap' and 'preprocessed' in cur_dirs):
                    for file_name in file_names:
                        # Check in bap dataset if the file extension is '.vhdr'
                        if dataset_name == 'hbn' or (dataset_name == 'bap' and os.path.splitext(file_name)[1] == ".vhdr"):
                            # Get the full path of the current file
                            file_path = os.path.join(dir_path, file_name)
                            # Create the new directory path for the output file
                            new_dir_path = os.path.join('/', *output_data_dirs, *cur_dirs[len(input_data_dirs):])

                            new_dir_path = os.path.abspath(new_dir_path)
                            os.makedirs(new_dir_path, exist_ok=True)
                            # Create the new file path for the output file with a '.npy' extension
                            new_file_path = os.path.join(new_dir_path, os.path.splitext(file_name)[0] + '.pickle')
                            pool.apply_async(preprocess_file, args=(file_path, dataset_name, hbn_formatter, new_file_path, sfreq, references, filters), callback=callback)

        # Wait for all processes to finish
        pool.close()
        pool.join()

    # Calculate the elapsed time in seconds
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


def preprocess_from_command_line():
    # Creating an argument parser object
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")

    # Adding command line arguments
    parser.add_argument("--datasets_path", default='/data0/practical-sose23/brain-age/data/', help="Path to the datasets directory")
    parser.add_argument("-dataset_names", nargs='+', help="List of dataset names")
    parser.add_argument("--downsampling", help="Factor for downsampling")
    parser.add_argument("--references", nargs='+', default=[], help="List of reference channels")
    parser.add_argument("--filters", default='{}', help="Optional key-value pairs in JSON format")
    
    # Parsing the command line arguments
    args = parser.parse_args()

    # Loading the JSON-formatted filters
    filters = json.loads(args.filters)
    
    # Preprocessing the data using the provided arguments
    preprocessed_data(args.datasets_path, args.dataset_names, args.downsampling, args.references, filters)


if __name__ == "__main__":
    preprocess_from_command_line()