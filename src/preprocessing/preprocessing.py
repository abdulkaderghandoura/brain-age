# Note: the "numexpr.utils - INFO" messages came from importing pandas in src/utils
# ---------------------------------------------------------------------------------

import os
import mne
import pickle
from tqdm import tqdm
import argparse
import warnings
import multiprocessing
from pathlib import Path

import sys
sys.path.append('../utils/')
from preprocessing_utils import get_hbn_formatter, apply_in_order, split_data
from preprocessing_utils import delete_partitioning_dirs, pickle_to_np, delete_files_with_extension
from utils import Timer, add_labels_to_splits


def preprocess_file(args, file_path: str, new_file_path: str, dataset_name: str, filters: dict, hbn_formatter):
    """Preprocesses one EEG file and save the preprocessed data to the new file path.

    Args:
        args (object): The arguments containing the preprocessing information.
        file_path (str): The path to the input file.
        new_file_path (str): The path to save the preprocessed data.
        dataset_name (str): The name of the dataset.
        filters (dict): A dictionary of filters to apply during preprocessing.
        hbn_formatter (HBNFormatter): The formatter for HBN data.


    Returns:
        Path: The output file path only in case of an issue, None otherwise.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    eeg_obj = None
    
    if dataset_name == 'bap':
        # Load EEG data into an EEG raw object
        eeg_obj = mne.io.read_raw_brainvision(file_path, verbose=False, preload=True)
        # Drop the two misc channels 'LE' and 'RE' in the bap data
        eeg_obj = eeg_obj.drop_channels(['LE', 'RE'])
        # Set channel types for all channels to 'eeg'
        eeg_obj.set_channel_types({channel: 'eeg' for channel in eeg_obj.ch_names})
        eeg_obj = eeg_obj.set_montage(montage)

        preprocessing_steps = [x for x in args.preprocessing_steps if x != 'bad_ch']
        eeg_obj = apply_in_order(eeg_obj, preprocessing_steps, filters, args.sfreq)
        
    
    elif dataset_name == 'lemon':
        # Load EEG data into an EEG raw object
        try:
            eeg_obj = mne.io.read_raw_brainvision(file_path, verbose=False, preload=True)
        except FileNotFoundError:
            return new_file_path
        # Drop the misc channel 'VEOG' in the lemon data
        eeg_obj = eeg_obj.drop_channels(['VEOG'])
        # Set channel types for all channels to 'eeg'
        eeg_obj.set_channel_types({channel: 'eeg' for channel in eeg_obj.ch_names})
        eeg_obj = eeg_obj.set_montage(montage)

        preprocessing_steps = [x for x in args.preprocessing_steps if x != 'bad_ch']
        eeg_obj = apply_in_order(eeg_obj, preprocessing_steps, filters, args.sfreq)

    elif dataset_name == 'hbn':
        assert 'bad_ch' in args.preprocessing_steps
        try:
            eeg_obj = hbn_formatter(file_path, args.preprocessing_steps, filters, args.sfreq)
        except Exception:
            return new_file_path

    for ref_item in args.references:
        channel_names = eeg_obj.ch_names
        if ref_item == 'average':
            eeg_obj.set_eeg_reference(ref_channels="average")
            eeg_obj = eeg_obj.pick(picks=channel_names)
        elif ref_item in channel_names:
            eeg_obj.set_eeg_reference(ref_channels=ref_item)
            eeg_obj = eeg_obj.pick(picks=channel_names)
    
    # Save the filtered EEG data as a pickle file
    with open(new_file_path, mode='wb') as out_file:
        pickle.dump(eeg_obj, out_file, protocol=5)

    return new_file_path


def preprocessed_data(args, filters):
    """Generates preprocessed data of the given datasets based on the specified settings.

    Args:
        args (object): The arguments containing the preprocessing information.
        filters (dict): A dictionary of filters to apply during preprocessing.
    """
    # Disable RuntimeWarning messages
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    datasets_path = Path(args.datasets_path).resolve()

    hbn_formatter = get_hbn_formatter(datasets_path)

    raw_directory_name = {'hbn': 'unzipped',
                          'bap': 'raw',
                          'lemon': 'LEMON_RAW'}
    
    g_file_paths = list()
    
    for dataset_name in args.dataset_names:
            
        input_data_path = datasets_path / dataset_name / raw_directory_name[dataset_name]
        output_data_path = datasets_path / dataset_name / 'preprocessed' / args.d_version
        
        file_paths = None
        if dataset_name == 'hbn':
            file_paths = list(input_data_path.rglob('*.mat'))
        elif dataset_name == 'bap':
            file_paths = [x for x in list(input_data_path.rglob('*.vhdr')) if 'preprocessed' in list(x.parts)]
        elif dataset_name == 'lemon':
            file_paths = list(input_data_path.rglob('*.vhdr'))

        for file_path in file_paths:
            cur_dirs = file_path.parent.parts
            new_dir_path = output_data_path.joinpath(*cur_dirs[len(input_data_path.parts):])
            os.makedirs(new_dir_path, exist_ok=True)
            new_file_path = new_dir_path / (file_path.stem + '.pickle')
            g_file_paths.append((file_path, new_file_path, dataset_name))
                
    unsaved_files = list()
    # Create a tqdm progress bar
    p_bar = tqdm(total=len(g_file_paths))
    # Set the number of worker processes (adjust according to your system)
    num_processes = min(15, multiprocessing.cpu_count())
    # Create a lock object to synchronize access to shared resources
    lock = multiprocessing.Lock()

    with multiprocessing.Pool(processes=num_processes) as pool:
        for g_file_path in g_file_paths:
            file_path, new_file_path, dataset_name = g_file_path
            result = pool.apply_async(preprocess_file, args=(args, file_path, new_file_path, 
                                                             dataset_name, filters, hbn_formatter))
    
            # Wait for the task to complete
            result.wait()
            new_file_path=result.get()

            if not new_file_path.is_file():
                print(f"Couldn't save: {new_file_path}")
                unsaved_files.append(new_file_path)
            with lock:
                # Manually increment the progress bar by 1
                p_bar.update(1)

        pool.close()
        pool.join()
        p_bar.close()
    
    # Report the number of problematic files and thir paths 
    for unsaved_file in unsaved_files:
        print(unsaved_file)
    print(f"Couldn't save the following {len(unsaved_files)} files:")


def main(args):
    timer = Timer()
    timer.start()

    if args.delete_ext is not None:
        print(f"Started deleting '.{args.delete_ext[1]}' extensions from {args.delete_ext[0]}")
        assert len(args.delete_ext) == 2 and Path(args.delete_ext[0]).is_dir()
        delete_files_with_extension(args.delete_ext[0], args.delete_ext[1])
    
    if args.delete_epochs:
        print('Started deleting epochs')
        delete_partitioning_dirs(args)

    filters = dict()
    for filter_name, filter_value in args.filter:
        # lpf: low-pass filter; hpf: high-pass filter; nf: notch filter
        assert filter_name in ['lpf', 'hpf', 'nf']
        filters[filter_name] = float(filter_value)

    print('Started filtering data')
    preprocessed_data(args, filters)

    split_ratios = {}
    for split in args.split:
        dataset_name = split[0]
        train_ratio, val_ratio, test_ratio = map(int, split[1:])
        split_ratios[dataset_name] = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}

    if args.eeg_partition_mode == 'numpy':
        pickle_to_np(args)

    print('Started splitting data')
    split_data(args, split_ratios)
    print('Started adding labels')
    add_labels_to_splits(args)
    
    # Print the elapsed time in seconds
    timer.end()


if __name__ == "__main__":
    # Creating an argument parser object
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    # List of all datasets can be preprocessed
    dataset_names = ['bap', 'lemon', 'hbn']
    # Default order of preprocessing steps
    pre_steps = ['lpf', 'hpf', 'sfreq', 'bad_ch']

    # Adding command line arguments
    parser.add_argument("--datasets_path", type=str, default='/data0/practical-sose23/brain-age/data/', 
                        help="Path to the datasets directory")
    
    parser.add_argument("--dataset_names", type=str, nargs='+', choices=dataset_names, required=True, 
                        help="List of dataset names")
    
    parser.add_argument("--d_version", type=str, required=True, 
                        help="Directory name of the current version of preprocessing")
    
    parser.add_argument("--sfreq", type=int, default=100, 
                        help="Sampling frequency")
    
    parser.add_argument('--filter', nargs='+', action='append', required=True,
                        help='Specify the filter and value, e.g., lpf 45')
    
    parser.add_argument("--references", type=str, nargs='+', default=[],
                        help="List of reference channels")
    
    parser.add_argument("--preprocessing_steps", type=str, nargs='+', default=pre_steps, choices=pre_steps, 
                        help='Preprocessing steps in order')
    
    parser.add_argument('--split', nargs='+', action='append', required=True,
                        help='Specify dataset and split ratios, e.g., bap 70 15 15')
    
    parser.add_argument("--window_in_sec", type=int, default=30, 
                        help='Window length in seconds for creating fixed lenth epochs')
    
    parser.add_argument("--stride_in_sec", type=int, default=15, 
                        help='Stride length in seconds for creating fixed lenth epochs')
    
    parser.add_argument("--eeg_partition_mode", type=str, default='mne', choices=['mne', 'numpy'],
                        help='The mode "mne" is to use a built-in function that drops bad epochs, \
                            resulting in less EEG data compared with the general "numpy" mode')
    
    parser.add_argument("--delete_epochs", action='store_true',
                        help='Deletes the directories that contains the fixed lengh epochs \
                              (NumPy parts of EEG) for the given dataset names and preprocessing version')
    
    parser.add_argument("--delete_ext", type=str, nargs='+', 
                        help='Delete files of the given extension, e.g., /path/to/dir/ pickle')

    args = parser.parse_args()
    main(args)
