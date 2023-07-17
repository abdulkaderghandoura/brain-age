import os
import random
import shutil
from pathlib import Path
import numpy as np
import mne
import pickle
from tqdm import tqdm

import scipy
from pyprep.find_noisy_channels import NoisyChannels


class InterpolateElectrodes:
    """
    interpolates between electrodes by recomputing the interpolation matrix for each sample
    """
    
    def __init__(self, from_montage, to_montage, to_channel_ordering, chs_to_exclude):

        # Get interpolation matrix given several mne montage (covering all channels of interest)
        self.chs_to_exclude = chs_to_exclude
            
        self.from_ch_pos = np.stack([value for key, value in from_montage.get_positions()["ch_pos"].items()
                                    if not key in self.chs_to_exclude])
        
        ch_name_to_pos = to_montage.get_positions()["ch_pos"]
        self.to_ch_pos = np.stack([ch_name_to_pos[ch_name] for ch_name in to_channel_ordering])
        
        self.interpolation_matrix = mne.channels.interpolation._make_interpolation_matrix(
                                    self.from_ch_pos, self.to_ch_pos)
    
    def __call__(self, x):
        x_interpolated = np.matmul(self.interpolation_matrix, x)
        return x_interpolated


class HBNFormatter:
    """Class for creating HBN EEG Raw objects with interpolated bad channels."""
    def __init__(self, bap_channel_names, montage_bap, montage_hbn, interpolation=True):
        
        self.montage_bap = montage_bap 
        self.montage_hbn = montage_hbn
        self.bap_channel_names = bap_channel_names
        self.ch_names_hbn = [ch for ch in self.montage_hbn.ch_names if "Fid" not in ch]
        self.interpolation = interpolation
        
        # Exclude duplicate locations and fiducials from interpolation
        chs_to_exclude = list(set(montage_bap.ch_names).difference(set(bap_channel_names)))
        chs_to_exclude = chs_to_exclude + [ch for ch in self.montage_hbn.ch_names if "Fid" in ch]
        # Set-up interpolation
        self.interpolate = InterpolateElectrodes(montage_hbn, montage_bap, bap_channel_names, chs_to_exclude)
        
        
    def __call__(self, mat_file, preprocessing_steps, filters, sfreq):
        # Load the .mat file
        mat_struct = scipy.io.loadmat(mat_file)

        # organize the meta data
        raw_data = mat_struct["EEG"]["data"][0][0]
        fs = mat_struct["EEG"]["srate"]
        
        # initialize raw hbn
        ch_types_hbn = len(self.ch_names_hbn)* ["eeg"]
        info = mne.create_info(ch_names=self.ch_names_hbn, sfreq=fs, ch_types=ch_types_hbn)
        info.set_montage(self.montage_hbn)
        raw = mne.io.RawArray(raw_data, info)

        bad_ch_idx = preprocessing_steps.index('bad_ch')
        raw = apply_in_order(raw, preprocessing_steps[:bad_ch_idx], filters, sfreq)
        
        # Find & repair bad channels (takes most of the processing time)
        nc = NoisyChannels(raw)
        nc.find_all_bads()
        raw.info['bads'].extend(nc.get_bads())
        raw.interpolate_bads(reset_bads=True)
        
        if self.interpolation:
            ch_types_bap = len(self.bap_channel_names)*["eeg"]
            info = mne.create_info(ch_names=self.bap_channel_names, sfreq=fs, ch_types=ch_types_bap)
            info.set_montage(self.montage_bap)
            raw_data = self.interpolate(raw._data)
            # Overwrite as raw bap
            raw = mne.io.RawArray(raw_data, info)
        
        # Apply the rest filters after bad_ch if any
        if bad_ch_idx < len(preprocessing_steps) - 1:
            raw = apply_in_order(raw, preprocessing_steps[bad_ch_idx + 1:], filters, sfreq)
        return raw


def get_hbn_formatter(datasets_path):
    """Creates an instance of HBNFormatter for formatting HBN data.

    Args:
        datasets_path (str or Path): The path to the datasets directory.

    Returns:
        HBNFormatter: An instance of HBNFormatter.
    """
    montage_bap = mne.channels.make_standard_montage("standard_1020")
    
    datasets_path = Path(datasets_path).resolve()
    custom_montage_path = datasets_path / 'hbn' / 'raw' / 'GSN_HydroCel_129.tsv'
    
    montage_hbn = mne.channels.read_custom_montage(custom_montage_path)
    
    with open(datasets_path / 'bap' / 'bap-channel-names.txt', 'r') as in_file:
        bap_channel_names = in_file.read().splitlines()
    
    bap_channel_names = [x for x in bap_channel_names if x not in ['LE', 'RE']]
    return HBNFormatter(bap_channel_names=bap_channel_names, montage_bap=montage_bap, montage_hbn=montage_hbn)


def apply_in_order(eeg_obj, preprocessing_steps, filters, sfreq):
    """Applies preprocessing steps to the EEG data in a specified order.

    Args:
        eeg_obj (object): The EEG object to apply the preprocessing steps to.
        preprocessing_steps (list): List of preprocessing steps to apply in order.
        filters (dict): Dictionary of filters to use as per the preprocessing step order.
        sfreq (int): The target sampling frequency.

    Returns:
        object: The preprocessed EEG object.
    """
    # Applying a band-pass filter in one step might be better
    if 'lpf' in preprocessing_steps and 'hpf' in preprocessing_steps:
        eeg_obj = eeg_obj.filter(l_freq=filters['hpf'], h_freq=filters['lpf'], h_trans_bandwidth=5, verbose=False)
        preprocessing_steps = [x for x in preprocessing_steps if x not in ['lpf', 'hpf']]
    
    for preprocessing_step in preprocessing_steps:
        if preprocessing_step == 'nf':
            assert 'nf' in filters
            # Apply a notch filter to the raw data
            eeg_obj = eeg_obj.notch_filter(freqs=filters['nf'], notch_widths=0.5, method='spectrum_fit', verbose=False)
        
        if preprocessing_step == 'lpf':
            assert 'lpf' in filters
            # Apply a low-pass filter to the raw data
            eeg_obj = eeg_obj.filter(l_freq=None, h_freq=filters['lpf'], h_trans_bandwidth=5, verbose=False)
        
        if preprocessing_step == 'hpf':
            assert 'hpf' in filters
            # Apply a high-pass filter to the raw data
            eeg_obj = eeg_obj.filter(l_freq=filters['hpf'], h_freq=None, verbose=False)

        if preprocessing_step == 'sfreq':
            assert sfreq is not None
            eeg_obj = eeg_obj.resample(sfreq=sfreq, verbose=False)
    
    return eeg_obj


def pickle_to_np(args):
    """Converts pickled EEG data to NumPy arrays.

    Args:
        args (object): The arguments containing the preprocessing information.
    """
    datasets_path = Path(args.datasets_path).resolve()
    pickle_file_paths = list()

    for dataset_name in args.dataset_names:
        input_data_path = datasets_path / dataset_name / 'preprocessed' / args.d_version
        pickle_file_paths.extend(list(input_data_path.rglob('*.pickle')))

    for file_path in tqdm(pickle_file_paths):
        with open(file_path, mode='rb') as in_file:
            eeg_obj = pickle.load(in_file)
            new_file_path = file_path.parent / (file_path.stem + '.npy')
            np.save(new_file_path , eeg_obj.get_data())


def delete_files_with_extension(directory_path, extension):
    """Deletes files with a specific extension in a directory and its subdirectories.

    Args:
        directory_path (str or Path): The path to the directory.
        extension (str): The file extension to delete (e.g., 'pickle').
    """
    # Delete leading dots if any
    extension = extension.lstrip('.')
    directory_path = Path(directory_path).resolve()
    file_paths = list(directory_path.rglob('*.' + extension))
    for file_path in file_paths:
        os.remove(file_path)
        print(f"Deleted: {file_path}")


def delete_partitioning_dirs(args):
    """Deletes the directories that contains the fixed lengh epochs (NumPy parts of EEG)
    for the given dataset names and preprocessing version.

    Args:
        args (object): The arguments containing the preprocessing information.
    """
    datasets_path = Path(args.datasets_path).resolve()
    dir_paths = None

    for dataset_name in tqdm(args.dataset_names):
        dataset_path = datasets_path / dataset_name / 'preprocessed' / args.d_version
        # Directories have a sibling pickled EEG object of the same name
        dir_paths = [x.parent / x.stem for x in list(dataset_path.rglob('*.pickle')) if (x.parent / x.stem).is_dir()]
    
    for dir_path in tqdm(dir_paths):
        shutil.rmtree(dir_path)
        print(f'Deleted: {dir_path}')


def partition_and_save(args, file_paths, out_file_path):
    """Create fixed-length EEG segments (aka epochs) and save their paths in the output text file. 
    This function supports two modes: mne and numpy. The main difference is that the mne built-in 
    function drops bad epochs automatically, resulting in a fewer number of data instances 
    compared to the numpy mode.

    Args:
        args (object): The arguments containing the preprocessing information.
        file_paths (list): List of file paths to partition and save.
        out_file_path (str or Path): The path to the output file to store NumPy parts of EEG
    """
    sub_file_paths = list()
    
    for file_path in file_paths:
        sub_dir_path = file_path.parent / file_path.stem
        prefix_name = sub_dir_path.name
        os.makedirs(sub_dir_path, exist_ok=True)
        eeg_parts = list()

        if args.eeg_partition_mode == 'numpy':
            with open(file_path, 'rb') as in_file:
                eeg_npy = np.load(in_file)
                
                # Number of samples for the window and the stride
                window_len = args.sfreq * args.window_in_sec
                stride_len = args.sfreq * args.stride_in_sec

                n_samples = eeg_npy.shape[1]
                # Number of partitions (aka epochs) according to the given configuratoin
                num_epochs = (n_samples - window_len) // stride_len + 1

                for i in range(num_epochs):
                    start_idx = i * stride_len
                    end_idx = start_idx + window_len
                    eeg_parts.append(eeg_npy[:, start_idx:end_idx])
        
        elif args.eeg_partition_mode == 'mne':
            with open(file_path, mode='rb') as in_file:
                eeg_obj = pickle.load(in_file)
                
                eeg_epochs = mne.make_fixed_length_epochs(eeg_obj, duration=args.window_in_sec, 
                                                          overlap=args.window_in_sec - args.stride_in_sec, 
                                                          preload=True, verbose=False)
                
                num_epochs = eeg_epochs.get_data().shape[0]
                for i in range(num_epochs):
                    eeg_parts.append(eeg_epochs.get_data()[i, :, :])
        
        for i, eeg_part in enumerate(eeg_parts):
            cur_partition_path = sub_dir_path / f'{prefix_name}_{i}.npy'
            np.save(cur_partition_path, eeg_part)
            sub_file_paths.append(cur_partition_path)

    with open(out_file_path, 'w') as out_file:
        for sub_file_path in sub_file_paths:
            out_file.write(str(sub_file_path) + '\n')


def split_data(args, split_ratios):
    """Splits the data into train, validation, and test sets based on the given split ratios.


    Args:
        args (object): The arguments containing the preprocessing information.
        split_ratios (dict): Dictionary of split ratios for each dataset in args.dataset_names.
    """
    seed_value = 42
    random.seed(seed_value)

    datasets_path = Path(args.datasets_path).resolve()

    for dataset_name, splits in tqdm(split_ratios.items()):
        data_path = datasets_path / dataset_name / 'preprocessed' / args.d_version
        
        file_paths = None
        if args.eeg_partition_mode == 'numpy':
            file_paths = [x for x in list(data_path.rglob('*.npy')) if not (x.parent.parent / (x.parent.name + '.pickle')).is_file()]
        elif args.eeg_partition_mode == 'mne':
            file_paths = list(data_path.rglob('*.pickle'))

        p_train, p_val, p_test = splits['train'], splits['val'], splits['test']
        # Check if the percentages add up to 100
        assert p_train + p_val + p_test == 100

        # Calculate the number of files for each set
        n_files = len(file_paths)
        n_val   = n_files * p_val // 100
        n_test  = n_files * p_test // 100
        n_train = n_files - n_val - n_test
        assert n_train + n_val + n_test == n_files

        # Shuffle the list of paths (Subjects)
        random.shuffle(file_paths)

        # Split the paths into train, val, and test sets
        train_paths = file_paths[:n_train]
        val_paths = file_paths[n_train:n_train + n_val]
        test_paths = file_paths[n_train + n_val:]
        assert len(train_paths) + len(val_paths) + len(test_paths) == len(file_paths)

        train_split_path = data_path / f'{dataset_name}_train.txt'
        val_split_path = data_path / f'{dataset_name}_val.txt'
        test_split_path = data_path / f'{dataset_name}_test.txt'

        partition_and_save(args, train_paths, train_split_path)
        partition_and_save(args, val_paths, val_split_path)
        partition_and_save(args, test_paths, test_split_path)
