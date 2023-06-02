import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, dataset_names, splits, transforms, sfreq=135, len_in_sec=30):
        self.sfreq = sfreq
        self.len_in_sec = len_in_sec
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transforms = transforms

        assert all(split in ['train', 'val', 'test'] for split in splits)
        assert all(dataset_name in ['hbn', 'bap'] for dataset_name in dataset_names)

        file_paths = list()
        for dataset_name in dataset_names:
            dataset_path = os.path.join('/data0/practical-sose23/brain-age/data', dataset_name, 'preprocessed')
            for split in splits:
                split_path = os.path.join(dataset_path, dataset_name + '_{}'.format(split) + '.txt')
                with open(split_path, 'r') as in_file:
                    lines = in_file.readlines()
                file_paths.extend([line.strip() for line in lines])
    
        self.data_paths = file_paths


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]

        with open(data_path, 'rb') as in_file:
            eeg_npy = np.load(in_file)

        # data = eeg_npy[:, :self.sfreq * self.len_in_sec].astype(np.float32)
        data = eeg_npy.astype(np.float32)
        # data_with_channel = torch.unsqueeze(torch.tensor(data), 0)
        data = torch.unsqueeze(torch.tensor(data), 0)

        return self.transforms(data)


if __name__ == "__main__":
    eeg_dataset = EEGDataset(['bap', 'hbn'], ['train', 'val', 'test'])
    print(len(eeg_dataset))