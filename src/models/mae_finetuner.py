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

class MAE_Finetuner(pl.LightningModule):
    def __init__(self, pretrained_model):
        super(MAE_Finetuner, self).__init__()
        self.pretrained_model = pretrained_model
        self.linear = nn.LazyLinear(1)
        self.r2 = R2Score()
        
    def forward(self, eegs):
        # Forward pass through the pre-trained model
        features = self.pretrained_model(eegs)
        # Forward pass through the linear layer
        output = self.linear(features)
        return output
    
    def training_step(self, batch, batch_idx):
        eegs, age = batch
        logits = self(eegs)
        loss = nn.functional.l1_loss(logits, age)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        r2 = self.r2(logits, age)
        self.log('train_r2', r2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        eegs, age = batch
        logits = self(eegs)
        loss = nn.functional.l1_loss(logits, age)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        r2 = self.r2(logits, age)
        self.log('val_r2', r2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.00001)
        return optimizer


