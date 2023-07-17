import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
# import lightning.pytorch as pl
# from lightning.pytorch.callbacks import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from torcheeg.models import EEGNet
from torch.utils.data import DataLoader
import wandb

sys.path.append('../../src/utils')
from transforms import _compose, _randomcrop, totensor, \
channelwide_norm, channelwise_norm, _clamp, toimshape, \
_labelcenter, _labelnorm, _labelbin
from dataset import EEGDataset

def _score_r2(y_hat, y, y_var):
    """Calculates the R-squared score between predicted values and ground truth.

    Args:
        y_hat (torch.Tensor): Predicted values.
        y (torch.Tensor): Ground truth.
        y_var (float): Variance of the target variable.

    Returns:
        float: The R-squared score.
    """
    return  1 - torch.nn.functional.mse_loss(y.squeeze(), y_hat.squeeze()) / y_var

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

    # torch.set_float32_matmul_precision('medium')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    datasets_path = Path(args.datasets_path).resolve()
    metadata_path = datasets_path / args.dataset_name / f'{args.dataset_name}-metadata.csv'
    df_subj = pd.read_csv(metadata_path)

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

    mean_age = torch.tensor(round(df_subj["Age"].mean(), 3))

    randomcrop = partial(_randomcrop, seq_len=args.chunk_size)
    clamp = partial(_clamp, dev_val=20.0)
    labelcenter = partial(_labelcenter, mean_age=round(df_subj["Age"].mean(), 3))
    labelbin = partial(_labelbin, y_lower=mean_age)
    composed_transforms = partial(_compose, transforms=[randomcrop, channelwise_norm, clamp])

    train_dataset = EEGDataset(args.datasets_path, [args.dataset_name], ['train'], d_version=args.d_version, transforms=composed_transforms, oversample=False)
    val_dataset = EEGDataset(args.datasets_path, [args.dataset_name], ['val'], d_version=args.d_version, transforms=composed_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=15, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=15, drop_last=True)

    wandb.login()

    logger = pl.loggers.WandbLogger(project="brain-age", name=args.experiment_name, 
                                    save_dir="/data0/practical-sose23/brain-age", log_model=False)
    
    early_stop_callback = EarlyStopping(monitor="validation loss", min_delta=0.00, patience=50, verbose=False, mode="min")

    trainer = pl.Trainer(
        callbacks=[early_stop_callback], 
        max_epochs=500, 
        accelerator="gpu",
        precision="bf16",
        logger=logger)

    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()


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
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--F1', default=16, type=int)
    parser.add_argument('--F2', default=32, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--depth_multiplier', default=2, type=int)

    parser.add_argument('--chunk_size', default=4, type=int)
    parser.add_argument('--kernel_1', default=2, type=int)
    parser.add_argument('--kernel_2', default=8, type=int)

    args = parser.parse_args()
    args.chunk_size *= args.sfreq
    args.kernel_1 = args.sfreq // args.kernel_1
    args.kernel_2 = args.sfreq // args.kernel_2

    main(args)
