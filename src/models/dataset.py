import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, dataset_name, split, sfreq=135, len_in_sec=30):
        self.sfreq = sfreq
        self.len_in_sec = len_in_sec
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert split in ['train', 'val', 'test']
        dataset_path = os.path.join('/data0/practical-sose23/brain-age/data', dataset_name, 'preprocessed')
        split_path = os.path.join(dataset_path, dataset_name + '.txt')
        with open(split_path, 'r') as in_file:
            lines = in_file.readlines()
        self.data_paths = [line.strip() for line in lines]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]

        with open(data_path, 'rb') as in_file:
            eeg_npy = np.load(in_file)

        data = eeg_npy[:, :self.sfreq * self.len_in_sec].astype(np.float32)
        data_with_channel = torch.unsqueeze(torch.tensor(data), 0)
        self.data = torch.unsqueeze(data_with_channel, 0)
        return self.data[index].to(self.device)


# if __name__ == "__main__":
#     eeg_dataset = EEGDataset('bap', 'train')
#     print(len(eeg_dataset[0]))