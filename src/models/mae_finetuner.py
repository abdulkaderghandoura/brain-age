# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch.optim as optim
import lightning.pytorch as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch.nn as nn
import torch
from torchmetrics import R2Score
from mae_age_regressor import AgeRegressor

def get_encoder_checksum(encoder):
    checksum = 0
    for params in encoder.parameters():
        checksum += params.sum()
    return checksum

class MAE_Finetuner(pl.LightningModule):
    def __init__(self, pretrained_model, lr, mode="finetune_encoder"):
        super(MAE_Finetuner, self).__init__()
        self.pretrained_model = pretrained_model
        print(f"========= \n checksum inside finetuner: {get_encoder_checksum(self.pretrained_model.blocks)}")
        self.head = AgeRegressor(output_dim=1)
        self.lr = lr
        self.r2 = R2Score()
        self.mode = mode
        
    def forward(self, eegs):
        features, *_ = self.pretrained_model.forward_encoder(eegs, mask_ratio=0.0, set_masking_seed=False)
        output = self.head(features)
        return output
    
    def training_step(self, batch, batch_idx):
        eegs, age = batch
        logits = self(eegs).squeeze()
        loss = nn.functional.l1_loss(logits, age)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        r2 = self.r2(logits, age)
        self.log('train_r2', r2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        eegs, age = batch
        logits = self(eegs).squeeze()
        loss = nn.functional.l1_loss(logits, age)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        r2 = self.r2(logits, age)
        self.log('val_r2', r2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        if self.mode == "linear_probe":
            self.pretrained_model.eval()
            optimizer = optim.AdamW(self.head.parameters(), lr=self.lr)
        elif self.mode == "finetune_encoder":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        else:
            print("select a valid mode for finetuning: linear_probe, finetune_encoder")
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=40, max_epochs=400)
        return [optimizer], [lr_scheduler]


