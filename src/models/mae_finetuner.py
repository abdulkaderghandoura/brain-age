# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block
import lightning.pytorch as pl

from mae_age_regressor import AgeRegressor

class VisionTransformer(pl.LightningModule):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, state_dict, mode, max_epochs, lr, warmup_epochs=6, **kwargs):

        super(VisionTransformer, self).__init__()
        self.backbone = timm.models.vision_transformer.VisionTransformer(**kwargs)
        del self.backbone.head
        if state_dict:
            self.backbone.load_state_dict(state_dict, strict=False)
        self.head = AgeRegressor(input_dim=kwargs["embed_dim"], output_dim=1)
        self.mode = mode        
        self.max_epochs = max_epochs
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.r2 = R2Score()

    def forward(self, x):
        features = self.backbone.forward_features(x)
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
            self.backbone.eval()
            optimizer = optim.AdamW(self.head.parameters(), lr=self.lr)
        elif self.mode == "finetune_encoder":
            optimizer = optim.AdamW(self.parameters(),  lr=self.lr)
        elif self.mode == "finetune_final_layer":
            final_modules = list(self.backbone.modules())[-3:]
            params = []
            for module in final_modules:
                print("...preparing module for finetuning:")
                print(module)
                params += list(module.parameters())
            params += list(self.head.parameters())
            optimizer = optim.AdamW(params, lr=self.lr)
        else:
            print("select a valid mode for finetuning: linear_probe, finetune_encoder, finetune_final_layer")
        
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, 
        max_epochs=self.max_epochs, warmup_epochs=self.warmup_epochs)
        return [optimizer], [lr_scheduler]




