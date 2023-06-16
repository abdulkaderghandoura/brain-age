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
# from format_hbn import HBNFormatter
from preprocessing_utils import HBNFormatter, get_hbn_formatter, apply_in_order

# pbar = None


def preprocess_file(file_path, dataset_name, hbn_formatter, preprocessing_steps, new_file_path=None, sfreq=None, references=[], filters=[]):
    eeg_obj = None
    if dataset_name == 'bap':
        # Load EEG data into an EEG raw object
        eeg_obj = mne.io.read_raw_brainvision(file_path, verbose=False, preload=True)
        eeg_obj = eeg_obj.drop_channels(['LE', 'RE'])
        # Set channel types for all channels to 'eeg'
        eeg_obj.set_channel_types({channel: 'eeg' for channel in eeg_obj.ch_names})
        # # Correct channel types of LE and RE
        # eeg_obj.set_channel_types({'LE':'misc', 'RE':'misc'})

        preprocessing_steps = [x for x in preprocessing_steps if x != 'bad_ch']
        eeg_obj = apply_in_order(eeg_obj, preprocessing_steps, filters, sfreq)
    
    elif dataset_name == 'hbn':
        assert 'bad_ch' in preprocessing_steps
        eeg_obj = hbn_formatter(file_path, preprocessing_steps, filters, sfreq)

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
    
    if new_file_path is not None:
        # Save the filtered EEG data as a pickle file
        with open(new_file_path, mode='wb') as out_file:
            pickle.dump(eeg_obj, out_file, protocol=5)

    return ((eeg_obj.get_data(), None) if new_file_path is None else (None, new_file_path))


def callback(new_file_path):
    print(f"Saved: {new_file_path[1]}")
    # Update the progress bar
    # pbar.update(1)


def preprocessed_data(datasets_path, dataset_names, preprocessing_version, preprocessing_steps, sfreq, references, filters):
    start_time = time.time()

    # Disable RuntimeWarning messages
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Split datasets_path into directories
    datasets_path_dirs = datasets_path.split('/' if '/' in datasets_path else '\\')
    datasets_path_dirs = [dir for dir in datasets_path_dirs if dir != '']

    hbn_formatter = get_hbn_formatter(datasets_path)
    paths_list = list()

    # Set the number of worker processes (adjust according to your system)
    # num_processes = multiprocessing.cpu_count()
    num_processes = 32

    with multiprocessing.Pool(processes=num_processes) as pool:

        for dataset_name in dataset_names:
            print(f'Started processing {dataset_name} dataset..')
            
            input_data_dirs = None
            if dataset_name == 'hbn':
                input_data_dirs = datasets_path_dirs + ['hbn', 'unzipped']
            elif dataset_name == 'bap':
                input_data_dirs = datasets_path_dirs + ['bap', 'raw']
            
            input_data_path = os.path.join('/', *input_data_dirs)

            output_data_dirs = datasets_path_dirs + [dataset_name, 'preprocessed']
            output_data_dirs.append(preprocessing_version)
            
            # Walk through the directory tree starting from input_data_path
            for dir_path, _, file_names in os.walk(input_data_path):
                
                # Split current directory path into individual directories
                cur_dirs = dir_path.split('/' if '/' in dir_path else '\\')
                cur_dirs = [dir for dir in cur_dirs if dir != '']
                
                if dataset_name == 'hbn' or (dataset_name == 'bap' and 'preprocessed' in cur_dirs):
                    for file_name in file_names:
                        # Check in bap dataset if the file extension is '.vhdr'
                        if dataset_name == 'hbn' or (dataset_name == 'bap' and file_name.endswith(".vhdr")):
                            # Get the full path of the current file
                            file_path = os.path.join(dir_path, file_name)
                            # Create the new directory path for the output file
                            
                            new_dir_path = os.path.join('/', *output_data_dirs, *cur_dirs[len(input_data_dirs):])

                            new_dir_path = os.path.abspath(new_dir_path)
                            os.makedirs(new_dir_path, exist_ok=True)
                            # Create the new file path for the output file with a '.npy' extension
                            new_file_path = os.path.join(new_dir_path, os.path.splitext(file_name)[0] + '.pickle')
                            paths_list.append((file_path, new_file_path))
                            # print('file path: ', file_path)
                            # print('new file path: ', new_file_path)
            
            # Create a tqdm progress bar
            # pbar = tqdm(total=len(paths_list))

            for item in paths_list:
                pool.apply_async(preprocess_file, args=(item[0], dataset_name, hbn_formatter, 
                                 preprocessing_steps, item[1], sfreq, references, filters), callback=callback)

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
    parser.add_argument("-version", default='v1.0', help="Directory name of the current version of preprocessing")
    parser.add_argument("--downsampling", help="Factor for downsampling")
    parser.add_argument("--references", nargs='+', default=[], help="List of reference channels")
    parser.add_argument("--filters", default={}, help='Optional key-value pairs in JSON format')
    parser.add_argument("--preprocessing_steps", nargs='+', default=['lpf', 'sfreq', 'bad_ch', 'hpf'], help='Preprocessing steps in order')
    
    # Parsing the command line arguments
    args = parser.parse_args()

    # Loading the JSON-formatted filters
    filters = json.loads(args.filters)
    
    # Preprocessing the data using the provided arguments
    preprocessed_data(args.datasets_path, args.dataset_names, args.version, args.preprocessing_steps, args.downsampling, args.references, filters)


if __name__ == "__main__":
    preprocess_from_command_line()
