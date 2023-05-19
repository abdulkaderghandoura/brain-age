from torch.utils.data import Dataset
import torch 
class DummyDataset(Dataset):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data = torch.rand(1, 1, 65, 650)
        self.targets = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self.data[idx].to(device), self.targets[idx].to(device)
#         return self.data, self.target