import os
# Use only the allocated GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import numpy as np
# import torch.optim as optim
import lightning.pytorch as pl
import torch.nn as nn
from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import R2Score
# import torch
# from functools import partial
import timm.models.vision_transformer
# from timm.models.vision_transformer import PatchEmbed, Block
from mae_age_regressor import AgeRegressor

class VisionTransformer(pl.LightningModule):
    """Lightning module for finetuning a pretrained vision transformer

    Args:
        state_dict (_type_): state dictionary of the pretrained model. If None, parameters are reinitialized.
        mode (string): mode to train different parameter groups. One of: linear_probe, finetune_encoder, finetune_final_layer.
        max_epochs (int): maximum number of training epochs with which to configure the scheduler.
        lr (float): learning rate.
        warmup_epochs (int): number of warmup epochs of the linear warmup cosine annealing scheduler.
        **vit_kwargs: additional keyword arguments to intialize the vision transformer

    """
    def __init__(self, state_dict, mode, max_epochs, lr, warmup_epochs=6, **vit_kwargs):

        super(VisionTransformer, self).__init__()
        
        # initialize a vision transformer
        self.backbone = timm.models.vision_transformer.VisionTransformer(**vit_kwargs)
        
        # remove the head of the backbone
        del self.backbone.head
        if state_dict:
            # load the parameters of the pretrained model
            self.backbone.load_state_dict(state_dict, strict=False)
        
        # initialize a regressor that maps from the embedding space to age
        self.head = AgeRegressor(input_dim=vit_kwargs["embed_dim"], output_dim=1)
        
        # initialize the remaining attributes
        self.mode = mode        
        self.max_epochs = max_epochs
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.r2 = R2Score()

    def forward(self, x):
        # extract the embedding of the cls token as features
        features = self.backbone.forward_features(x)
        # regress age
        output = self.head(features)
        return output
    
    def training_step(self, batch, batch_idx):
        eegs, age = batch
        # compute predictions and loss
        logits = self(eegs).squeeze()
        loss = nn.functional.l1_loss(logits, age)
        # log loss and metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        r2 = self.r2(logits, age)
        self.log('train_r2', r2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        eegs, age = batch
        # compute predictions and loss
        logits = self(eegs).squeeze()
        loss = nn.functional.l1_loss(logits, age)
        # log loss and metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        r2 = self.r2(logits, age)
        self.log('val_r2', r2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        if self.mode == "linear_probe":
            # freeze the parameters of the backbone
            self.backbone.eval()
            # train the regression head
            params = head.parameters()
        elif self.mode == "finetune_encoder":
            # train the entire model
            params = self.parameters()
        elif self.mode == "finetune_final_layer":
            # train the final layer (last 3 modules)
            final_layer = list(self.backbone.modules())[-3:]
            # loop over the modules of the final layer
            params = []
            for module in final_layer:
                print("...preparing module for finetuning:")
                print(module)
                # extract the module parameters for training
                params += list(module.parameters())
            # additionally, train the regression head
            params += list(self.head.parameters())
        else:
            print("select a valid mode for finetuning: linear_probe, finetune_encoder, finetune_final_layer")
        
        # expose the relevant parameters to the optimizer  
        optimizer = AdamW(params, lr=self.lr)
        
        # configure the learning rate scheduler
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, 
        max_epochs=self.max_epochs, warmup_epochs=self.warmup_epochs)
        
        return [optimizer], [lr_scheduler]




