import pickle
import time
import mne
import os
import random
import shutil
import numpy as np
from tqdm import tqdm
from format_hbn import HBNFormatter
from utils import get_dirs

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


def pickle_to_np(datasets_path, dataset_names):
    start_time = time.time()
    file_paths = list()

    # Split datasets_path into directories
    datasets_path_dirs = get_dirs(datasets_path)

    for dataset_name in dataset_names:
        input_data_path = os.path.join('/', *datasets_path_dirs, dataset_name, 'preprocessed')
        # Walk through the directory tree starting from input_data_path
        for dir_path, _, file_names in os.walk(input_data_path):
            for file_name in file_names:
                if file_name.endswith(".pickle"):
                    # Get the full path of the current file
                    file_path = os.path.join(dir_path, file_name)
                    file_paths.append(file_path)

    for item in tqdm(file_paths):
        with open(item, mode='rb') as in_file:
            eeg_obj = pickle.load(in_file)
            np.save(os.path.splitext(item)[0] + '.npy', eeg_obj.get_data())

    # Calculate the elapsed time in seconds
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    

def delete_partition_directories(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npy'):
                file_path = os.path.join(dirpath, filename)
                dir_path = os.path.splitext(file_path)[0]
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"Deleted sibling directory: {dir_path}")


def partition_and_save(file_paths, out_file_path, part_len):
    sub_file_paths = list()
    for file_path in tqdm(file_paths):
        sub_dir_path = os.path.splitext(file_path)[0]
        prefix_name = os.path.basename(os.path.normpath(sub_dir_path))
        os.makedirs(sub_dir_path, exist_ok=True)
        with open(file_path, 'rb') as in_file:
            eeg_npy = np.load(in_file)

            n_samples = eeg_npy.shape[1]
            n_partitions = n_samples // part_len

            for i in range(n_partitions):
                start_idx = i * part_len
                end_idx = start_idx + part_len
                eeg_part = eeg_npy[:, start_idx:end_idx]
                cur_partition_path = os.path.join(sub_dir_path, f"{prefix_name}_{i}.npy")
                np.save(cur_partition_path, eeg_part)
                sub_file_paths.append(cur_partition_path)

    with open(out_file_path, 'w') as out_file:
        for sub_file_path in sub_file_paths:
            out_file.write(sub_file_path + '\n')


def split_data(datasets_path, config, sfreq=135, len_in_sec=30):
    start_time = time.time()
    # Split datasets_path into directories
    datasets_path_dirs = get_dirs(datasets_path)

    for dataset_name, splits in config.items():
        
        input_data_path = os.path.join('/', *datasets_path_dirs, dataset_name, 'preprocessed')
        file_paths = list()

        # Walk through the directory tree starting from input_data_path
        for dir_path, _, file_names in os.walk(input_data_path):
            dir_path_dirs = get_dirs(dir_path)
            if not os.path.exists(os.path.join(*dir_path_dirs[:-1], dir_path_dirs[-1] + '.npy')):
                for file_name in file_names:
                    if file_name.endswith(".npy"):
                        # Get the full path of the current file
                        file_path = os.path.join(dir_path, file_name)            
                        file_paths.append(file_path)

        p_train, p_val, p_test = splits['train'], splits['val'], splits['test']
        # Check if the percentages add up to 100
        assert p_train + p_val + p_test == 100.0

        # Calculate the number of files for each set
        n_files = len(file_paths)
        n_train = int((p_train / 100) * n_files)
        n_val   = int((p_val / 100) * n_files)
        n_test  = int((p_test / 100) * n_files)

        # Set the seed
        seed_value = 42
        random.seed(seed_value)

        # Shuffle the list of paths
        random.shuffle(file_paths)

        # Split the paths into train, val, and test sets
        train_paths = file_paths[:n_train]
        val_paths = file_paths[n_train:n_train + n_val]
        test_paths = file_paths[n_train + n_val:]

        # Save the sets as text files
        train_split_path = os.path.join(input_data_path, f"{dataset_name}_train.txt")
        val_split_path = os.path.join(input_data_path, f"{dataset_name}_val.txt")
        test_split_path = os.path.join(input_data_path, f"{dataset_name}_test.txt")

        partition_and_save(train_paths, train_split_path, sfreq * len_in_sec)
        partition_and_save(val_paths, val_split_path, sfreq * len_in_sec)
        partition_and_save(test_paths, test_split_path, sfreq * len_in_sec)


    # Calculate the elapsed time in seconds
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    config = {'hbn':{
                        'train': 60,
                        'val': 20,
                        'test': 20
                    },
              'bap':{
                        'train': 70,
                        'val': 15,
                        'test': 15
                    }
             }
    split_data('/data0/practical-sose23/brain-age/data', config)

    # root_directory = '/data0/practical-sose23/brain-age/data/bap/preprocessed'
    # delete_partition_directories(root_directory)

#     pickle_to_np('/data0/practical-sose23/brain-age/data', ['hbn', 'bap'])