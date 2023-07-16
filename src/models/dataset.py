import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class EEGDataset(Dataset):
    def __init__(self, datasets_path, dataset_names, splits, d_version, transforms, oversample=False, labelcenter=True):
        datasets_path = Path(datasets_path).resolve()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transforms = transforms

        assert all(split in ['train', 'val', 'test'] for split in splits)
        assert all(dataset_name in ['hbn', 'bap'] for dataset_name in dataset_names)

        file_paths = {}
        age_dict = {}

        for dataset_name in dataset_names:
            dataset_path = datasets_path / dataset_name / 'preprocessed' / d_version
            file_paths[dataset_name] = {}
            age_dict[dataset_name] = {}
            for split in splits:
                split_path = dataset_path / f'{dataset_name}_{split}.csv'
                data = np.loadtxt(split_path, dtype=str ,delimiter=',',skiprows=1)
                file_paths[dataset_name][split] = data[:, 0]
                age_dict[dataset_name][split] = data[:, 1].astype(float)

        if labelcenter and len(dataset_names) == 1:
            for split in splits:
                # print('**************', age_dict[dataset_names[0]][split])
                mean_age = np.round(np.mean(age_dict[dataset_names[0]][split]), 3)
                age_dict[dataset_names[0]][split] -= mean_age

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

        self.target = torch.tensor(np.round((np.array(age)).astype(float), 3))
        self.data_paths = data_paths
        
        seed_value = 42
        np.random.seed(seed_value)
        rand_permutation = np.random.permutation(len(self.target))
        self.target = self.target[rand_permutation]
        self.data_paths = self.data_paths[rand_permutation]


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        target = self.target[index]


        with open(data_path, 'rb') as in_file:
            eeg_npy = np.load(in_file)

        data = eeg_npy.astype(np.float32)
        data = torch.unsqueeze(torch.tensor(data), 0)
        
        return self.transforms(data), target
