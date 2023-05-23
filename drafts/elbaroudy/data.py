from torch.utils.data import Dataset
import torch 
import pickle 
import mne 
import numpy as np 
montage = mne.channels.make_standard_montage('standard_1020')
class DummyDataset(Dataset):
    def __init__(self):
        file_path = "/data0/practical-sose23/brain-age/filtered_data/healthy_controls/preprocessed/Exp_eyes_closed_vpH39_eyes_closed.pickle"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(file_path, mode='rb') as in_file:
            raw_filtered = pickle.load(in_file)
        raw_filtered.set_montage(montage, on_missing='ignore')
        data = raw_filtered.get_data()[:, :4000].astype(np.float32)
        data_with_channel = torch.unsqueeze(torch.tensor(data), 0)
        self.data = torch.unsqueeze(data_with_channel, 0)
#         self.data = torch.rand(1, 1, 65, 650)
        self.targets = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self.data[idx].to(device), self.targets[idx].to(device)
#         return self.data, self.target