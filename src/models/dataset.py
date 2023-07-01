import os
import numpy as np
import torch
from torch.utils.data import Dataset
import csv 

class EEGDataset(Dataset):
    def __init__(self, dataset_names, splits, transforms, sfreq=135, len_in_sec=30, oversample=False):
        self.sfreq = sfreq
        self.len_in_sec = len_in_sec
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transforms = transforms

        assert all(split in ['train', 'val', 'test'] for split in splits)
        assert all(dataset_name in ['hbn', 'bap', 'lemon'] for dataset_name in dataset_names)

        # file_paths = np.array([])
        # age = np.array([])
        file_paths = {}
        age_dict = {}

        for dataset_name in dataset_names:
            dataset_path = os.path.join('/data0/practical-sose23/brain-age/data', dataset_name, 'preprocessed/v2.0')
            file_paths[dataset_name] = {}
            age_dict[dataset_name] = {}
            for split in splits:
                split_path = os.path.join(dataset_path, dataset_name + '_{}'.format(split) + '.csv')
                data = np.loadtxt(split_path, dtype=str ,delimiter=',',skiprows=1)
                file_paths[dataset_name][split] = data[:, 0]
                age_dict[dataset_name][split] = data[:, 1]
                
                # file_paths.extend(data[:, 0])

                # age = np.concatenate((age, data[:, 1]))
                # file_paths.extend([line.strip() for line in lines])

        if oversample and 'train' in split and len(dataset_names) > 1: 
            minority_len = len(file_paths['bap']['train'])
            majority_len = len(file_paths['hbn']['train'])

            ratio = majority_len // minority_len
            num_to_oversample = int(minority_len * (ratio - 1))

            oversample_indices = np.random.choice(np.arange(0, minority_len, 1), size=num_to_oversample)
            oversampled_data_path = file_paths['bap']['train'][oversample_indices]
            oversampled_age = age_dict['bap']['train'][oversample_indices]

            file_paths['bap']['train'] = np.concatenate((file_paths['bap']['train'], oversampled_data_path))
            age_dict['bap']['train'] = np.concatenate((age_dict['bap']['train'], oversampled_age),)
        
        age = np.array([])
        data_paths = np.array([])
        for data_name, splits in file_paths.items():
            for split, data in splits.items(): 
                data_paths = np.concatenate((data_paths, file_paths[data_name][split]))
                age = np.concatenate((age, age_dict[data_name][split]))

        self.target = torch.tensor(np.round((np.array(age)).astype(float)).astype(int))
        self.data_paths = data_paths


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        target = self.target[index]

        with open(data_path, 'rb') as in_file:
            eeg_npy = np.load(in_file)

        # data = eeg_npy[:, :self.sfreq * self.len_in_sec].astype(np.float32)
        data = eeg_npy.astype(np.float32)
        # data_with_channel = torch.unsqueeze(torch.tensor(data), 0)
        data = torch.unsqueeze(torch.tensor(data), 0)

        return self.transforms(data), target


if __name__ == "__main__":
    eeg_dataset = EEGDataset(['bap', 'hbn'], ['train', 'val', 'test'])
    print(len(eeg_dataset))