import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        eeg_data = np.load(data_path)
        tensor_data = torch.from_numpy(eeg_data).float()
        return tensor_data
