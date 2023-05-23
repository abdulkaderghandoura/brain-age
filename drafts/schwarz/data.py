from torch.utils.data import Dataset

class BrainAgeDataset(Dataset):
    def __init__(self, epochs, ages, transforms=lambda x:x, target_transforms=lambda x:x):
        self.epochs = epochs
        self.ages = ages
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, idx):
        return self.transforms(self.epochs[idx]), self.target_transforms(self.ages[idx])
    
    def __len__(self):
        return len(self.ages)
    
   