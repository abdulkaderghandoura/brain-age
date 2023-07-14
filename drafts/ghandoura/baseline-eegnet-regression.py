import argparse
import numpy as np
import pandas as pd
# import sklearn
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import mne
import pathlib
import lightning.pytorch as pl
import torch
# import torcheeg
# import xgboost
import wandb
# import autoreject
# from tqdm.notebook import tqdm
from torcheeg.models import EEGNet
from functools import partial
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
# import csv
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

import sys
sys.path.append('../../src/utils')
# from transforms import channelwide_norm, channelwise_norm, _clamp, _randomcrop, _compose
from transforms import _compose, _randomcrop, totensor, \
channelwide_norm, channelwise_norm, _clamp, toimshape, \
_labelcenter, _labelnorm, _labelbin

def _score_r2(y_hat, y, y_var):
    return  1 - torch.nn.functional.mse_loss(y.squeeze(), y_hat.squeeze()) / y_var

class EEGDataset(Dataset):
    def __init__(self, dataset_names, splits, d_version, transforms, sfreq=135, len_in_sec=30, oversample=False):
        self.sfreq = sfreq
        self.len_in_sec = len_in_sec
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transforms = transforms

        assert all(split in ['train', 'val', 'test'] for split in splits)
        assert all(dataset_name in ['hbn', 'bap', 'lemon'] for dataset_name in dataset_names)

        file_paths = {}
        age_dict = {}

        for dataset_name in dataset_names:
            dataset_path = os.path.join('/data0/practical-sose23/brain-age/data', dataset_name, 'preprocessed', d_version)
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
        # &&&&&&&&&&&&
        # data = torch.unsqueeze(torch.tensor(data), 0)
        data = torch.tensor(data)

        return self.transforms(data), target


class BrainAgeModel(pl.LightningModule):
        
    def __init__(self, model, args, loss_func, metric=None):
        super().__init__()
        self.model = model
        self.args = args
        self.loss_func = loss_func
        self.metric = metric
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # print("&&&&&&&&&&&&&&&&&&&&&: ", x.shape)
        y_hat = self.forward(x)
        val_loss = self.loss_func(y_hat.squeeze(), y.squeeze())
        self.log("validation loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.metric:
            metric_val = self.metric["func"](y_hat.squeeze(), y.squeeze())
#             print(f"==================Val Acc: {metric_val} =======================")
#             print(y_hat.squeeze()[:2], y.squeeze()[:2])
            self.log("val_"+self.metric["name"], metric_val, on_step=True, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat.squeeze(), y.squeeze())
        self.log("training loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.metric:
            metric_val = self.metric["func"](y_hat.squeeze(), y.squeeze())
#             print(f"==================Train Acc: {metric_val} =======================")
#             print(y_hat.squeeze()[:2], y.squeeze()[:2])
            self.log("train_"+self.metric["name"], metric_val, on_step=True, on_epoch=True, prog_bar=True)

        return loss


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.set_float32_matmul_precision('medium')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    metadata_path = os.path.join(args.datasets_path, args.dataset_name, f'{args.dataset_name}-metadata.csv')
    df_subj = pd.read_csv(metadata_path)
    
    print('Step 1: Done!')
    # --------------------------------------------

    eegnet = EEGNet(chunk_size=args.chunk_size,
                    num_electrodes=args.num_electrodes,
                    dropout=args.dropout,
                    kernel_1=args.kernel_1,
                    kernel_2=args.kernel_2,
                    F1=args.F1,
                    F2=args.F2,
                    D=args.depth_multiplier,
                    num_classes=1)

    score_r2 = partial(_score_r2, y_var=df_subj["Age"].var())

    model = BrainAgeModel(model=list(eegnet.modules())[0], 
                         args=args, 
                         loss_func=torch.nn.functional.l1_loss,
                         metric={"name":"r2", "func":score_r2})
    
    print(model)

    print('Step 2: Done!')
    # --------------------------------------------
    # print('********************: ', args.chunk_size)
    mean_age = torch.tensor(round(df_subj["Age"].mean(), 3))

    randomcrop = partial(_randomcrop, seq_len=args.chunk_size)
    # &&&&&&&&&&&&&&&
    clamp = partial(_clamp, dev_val=20.0)
    # clamp = partial(_clamp, dev_val=0.5)
    labelcenter = partial(_labelcenter, mean_age=round(df_subj["Age"].mean(), 3))
    labelbin = partial(_labelbin, y_lower=mean_age)
    # &&&&&&&&&&&&&&
    composed_transforms = partial(_compose, transforms=[randomcrop, channelwise_norm, clamp, toimshape])
    # composed_transforms = partial(_compose, transforms=[randomcrop, channelwise_norm, toimshape])
    # target_transforms = partial(_compose, transforms=[labelcenter, totensor])

    print('Step 3: Done!')
    # --------------------------------------------

    train_dataset = EEGDataset([args.dataset_name], ['train'], d_version=args.d_version, transforms=composed_transforms, oversample=False)
    val_dataset = EEGDataset([args.dataset_name], ['val'], d_version=args.d_version, transforms=composed_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=16, drop_last=True)

    print('Step 4: Done!')
    # --------------------------------------------

    wandb.login()

    logger = pl.loggers.WandbLogger(project="brain-age", name=args.experiment_name, 
                                    save_dir="/data0/practical-sose23/brain-age", log_model=False)
    # $$$$$$$$$$$$$$ patience=25
    early_stop_callback = EarlyStopping(monitor="validation loss", min_delta=0.00, patience=50, verbose=False, mode="max")

    trainer = pl.Trainer(
        callbacks=[early_stop_callback], 
        max_epochs=500, 
        accelerator="gpu",
        precision="bf16-mixed",
        logger=logger)

    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()

    print('Step 5: Done!')


if __name__ == "__main__":
    # Creating an argument parser object
    parser = argparse.ArgumentParser(description="Baseline EEGNet Regression")
    # List of all datasets can be preprocessed
    dataset_names = ['bap', 'hbn', 'lemon']

    # Adding command line arguments
    parser.add_argument("--datasets_path", type=str, default='/data0/practical-sose23/brain-age/data/', 
                        help="Path to the datasets directory")
    
    parser.add_argument("--dataset_name", type=str, choices=dataset_names, required=True, 
                        help="List of dataset names")

    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--d_version', type=str, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--sfreq', default=100, type=int)
    parser.add_argument('--num_electrodes', type=int, required=True,)

    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--F1', default=16, type=int)
    parser.add_argument('--F2', default=32, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--depth_multiplier', default=2, type=int)

    #&&&&&&&&&&&&&&&&&&&& TODO: read chunk_size in sec instead
    parser.add_argument('--chunk_size', default=2, type=int)
    parser.add_argument('--kernel_1', default=2, type=int)
    parser.add_argument('--kernel_2', default=8, type=int)

    args = parser.parse_args()
    args.chunk_size *= args.sfreq
    args.kernel_1 = args.sfreq // args.kernel_1
    args.kernel_2 = args.sfreq // args.kernel_2
    
    main(args)