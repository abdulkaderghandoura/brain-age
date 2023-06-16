import pickle
import time
import mne
import os
import random
import shutil
import numpy as np
from tqdm import tqdm
from utils import get_dirs

import scipy
import matplotlib.pyplot as plt
from pyprep.find_noisy_channels import NoisyChannels


class InterpolateElectrodes:
    """
    interpolates between electrodes by recomputing the interpolation matrix for each sample
    """
    
    def __init__(self, from_montage, to_montage, to_channel_ordering, chs_to_exclude):

        ### Get interpolation matrix given several mne montage (covering all channels of interest)
        self.chs_to_exclude = chs_to_exclude
            
        self.from_ch_pos = np.stack(
            [value for key, value in from_montage.get_positions()["ch_pos"].items() \
             if not key in self.chs_to_exclude]
        )
        
        ch_name_to_pos = to_montage.get_positions()["ch_pos"]
        self.to_ch_pos = np.stack(
            [ch_name_to_pos[ch_name] for ch_name in to_channel_ordering]
        )
        
        self.interpolation_matrix = mne.channels.interpolation._make_interpolation_matrix(
                self.from_ch_pos, self.to_ch_pos
                )
    def __call__(self, x):
        x_interpolated = np.matmul(self.interpolation_matrix, x)
        return x_interpolated

class HBNFormatter:
    
    def __init__(self, channel_names_bap, montage_bap, montage_hbn, interpolation=True):
        
        self.montage_bap = montage_bap 
        self.montage_hbn = montage_hbn
        self.channel_names_bap = channel_names_bap
        self.ch_names_hbn = [ch for ch in self.montage_hbn.ch_names if "Fid" not in ch]
        self.interpolation = interpolation
        
        # exclude duplicate locations and fiducials from interpolation
        chs_to_exclude = list(set(montage_bap.ch_names).difference(set(channel_names_bap)))
        chs_to_exclude = chs_to_exclude + [ch for ch in self.montage_hbn.ch_names if "Fid" in ch]
        # set-up interpolation
        self.interpolate = InterpolateElectrodes(montage_hbn, montage_bap, channel_names_bap, chs_to_exclude)
        
        
        
    def __call__(self, mat_file, preprocessing_steps, filters, sfreq):
        ## Load the .mat file
        mat_struct = scipy.io.loadmat(mat_file)

        ## organize the meta data
        raw_data = mat_struct["EEG"]["data"][0][0]
        fs = mat_struct["EEG"]["srate"]
        
        ## initialize raw hbn
        ch_types_hbn = len(self.ch_names_hbn)* ["eeg"]
        info = mne.create_info(ch_names=self.ch_names_hbn, sfreq=fs, ch_types=ch_types_hbn)
        info.set_montage(self.montage_hbn)
        raw = mne.io.RawArray(raw_data, info)

        bad_ch_idx = preprocessing_steps.index('bad_ch')
        eeg_obj = apply_in_order(raw, preprocessing_steps[:bad_ch_idx], filters, sfreq)
        
        ## find & repair bad channels (takes most of the processing time)
        nc = NoisyChannels(raw)
        nc.find_all_bads()
        raw.info['bads'].extend(nc.get_bads())
        raw.interpolate_bads(reset_bads=True)
        
        if self.interpolation:
            ch_types_bap = len(self.channel_names_bap)*["eeg"]
            info = mne.create_info(ch_names=self.channel_names_bap, sfreq=fs, ch_types=ch_types_bap)
            info.set_montage(self.montage_bap)
            raw_data = self.interpolate(raw._data)
            ## overwrite as raw bap
            raw = mne.io.RawArray(raw_data, info)

        return apply_in_order(raw, preprocessing_steps[bad_ch_idx + 1:], filters, sfreq)


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


def apply_in_order(eeg_obj, preprocessing_steps, filters={}, sfreq=None):
    for preprocessing_step in preprocessing_steps:
        if preprocessing_step == 'nf':
            assert 'nf' in filters
            # Apply a notch filter to the raw data
            eeg_obj = eeg_obj.notch_filter(freqs=filters['nf'], notch_widths=0.5, method='spectrum_fit', verbose=False)
        
        if preprocessing_step == 'lpf':
            assert 'lpf' in filters
            # Apply a low-pass filter to the raw data
            eeg_obj = eeg_obj.filter(l_freq=None, h_freq=filters['lpf'], fir_design='firwin', h_trans_bandwidth=5, verbose=False)
        
        if preprocessing_step == 'hpf' in filters:
            assert 'hpf' in filters
            # Apply a high-pass filter to the raw data
            eeg_obj = eeg_obj.filter(l_freq=filters['hpf'], h_freq=None, fir_design='firwin', h_trans_bandwidth=5, verbose=False)

        if preprocessing_step == 'sfreq':
            assert sfreq is not None
            eeg_obj = eeg_obj.resample(sfreq=sfreq, verbose=False)

    return eeg_obj


def pickle_to_np(datasets_path, dataset_names, d_version):
    start_time = time.time()
    file_paths = list()

    # Split datasets_path into directories
    datasets_path_dirs = get_dirs(datasets_path)

    for dataset_name in dataset_names:
        input_data_path = os.path.join(os.path.sep, *datasets_path_dirs, dataset_name, 'preprocessed', d_version)
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


def delete_files_with_extension(target_dir, extension):
    for dir_path, _, file_names in os.walk(target_dir):
        for file_name in file_names:
            if file_name.endswith(extension):
                file_path = os.path.join(dir_path, file_name)
                os.remove(file_path)
                print(f"Deleted: {file_path}")


def delete_partition_directories(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npy'):
                file_path = os.path.join(dirpath, filename)
                dir_path = os.path.splitext(file_path)[0]
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"Deleted sibling directory: {dir_path}")


def partition_and_save(file_paths, out_file_path, window_len, stride_len):
    sub_file_paths = list()
    for file_path in tqdm(file_paths):
        sub_dir_path = os.path.splitext(file_path)[0]
        prefix_name = os.path.basename(os.path.normpath(sub_dir_path))
        os.makedirs(sub_dir_path, exist_ok=True)
        with open(file_path, 'rb') as in_file:
            eeg_npy = np.load(in_file)

            n_samples = eeg_npy.shape[1]
            n_partitions = (n_samples - window_len) // stride_len + 1

            for i in range(n_partitions):
                start_idx = i * stride_len
                end_idx = start_idx + window_len
                eeg_part = eeg_npy[:, start_idx:end_idx]
                cur_partition_path = os.path.join(sub_dir_path, f"{prefix_name}_{i}.npy")
                np.save(cur_partition_path, eeg_part)
                sub_file_paths.append(cur_partition_path)

    with open(out_file_path, 'w') as out_file:
        for sub_file_path in sub_file_paths:
            out_file.write(sub_file_path + '\n')


def split_data(datasets_path, config, d_version, sfreq=135, window_in_sec=30, stride_in_sec=15):
    start_time = time.time()
    # Split datasets_path into directories
    datasets_path_dirs = get_dirs(datasets_path)

    for dataset_name, splits in config.items():
        
        input_data_path = os.path.join(os.path.sep, *datasets_path_dirs, dataset_name, 'preprocessed', d_version)
        file_paths = list()

        # Walk through the directory tree starting from input_data_path
        for dir_path, _, file_names in os.walk(input_data_path):
            dir_path_dirs = get_dirs(dir_path)
            if not os.path.exists(os.path.join(os.path.sep, *dir_path_dirs[:-1], dir_path_dirs[-1] + '.npy')):
                for file_name in file_names:
                    if file_name.endswith(".npy"):
                        # Get the full path of the current file
                        file_path = os.path.join(dir_path, file_name)            
                        file_paths.append(file_path)

        p_train, p_val, p_test = splits['train'], splits['val'], splits['test']
        # Check if the percentages add up to 100
        assert p_train + p_val + p_test == 100

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

        partition_and_save(train_paths, train_split_path, sfreq * window_in_sec, sfreq * stride_in_sec)
        partition_and_save(val_paths, val_split_path, sfreq * window_in_sec, sfreq * stride_in_sec)
        partition_and_save(test_paths, test_split_path, sfreq * window_in_sec, sfreq * stride_in_sec)


    # Calculate the elapsed time in seconds
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    config = {
                # 'hbn':{
                #         'train': 60,
                #         'val': 20,
                #         'test': 20
                # },
                'bap':{
                        'train': 70,
                        'val': 15,
                        'test': 15
                }
             }
    # split_data('/data0/practical-sose23/brain-age/data', config, 'v1.0')

    # root_directory = '/data0/practical-sose23/brain-age/data/bap/preprocessed/v1.0'
    # delete_partition_directories(root_directory)

    # pickle_to_np('/data0/practical-sose23/brain-age/data', ['bap'], 'v1.0')

    # delete_files_with_extension(root_directory, '.npy')