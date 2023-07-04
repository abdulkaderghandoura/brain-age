import wandb
from mae import MaskedAutoencoderViT
import lightning.pytorch as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch.nn as nn
from torch.optim import Adam, AdamW, LBFGS
import torch 

from matplotlib import pyplot as plt
# import numpy as np

from torchmetrics import R2Score
# class AgeRegressor(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(AgeRegressor, self).__init__()

#         hidden_size = input_dim // 2
#         self.linear_one = nn.Linear(input_dim, hidden_size)
#         self.batch_norm_one = nn.BatchNorm1d(hidden_size)
#         self.linear_two = nn.Linear(hidden_size, output_dim)
#         self.relu = nn.ReLU()


#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         age = self.linear_two(self.relu(self.batch_norm_one(self.linear_one(x))))
#         # age = self.linear(x)
#         return age
    
class AgeRegressor(nn.Module):
    def __init__(self, output_dim):
        super(AgeRegressor, self).__init__()
        
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.act(x)
        return x

class MAE_AGE(pl.LightningModule):
    def __init__(self, img_size=(63, 1000), patch_size=(1, 100), in_chans=1,
                embed_dim=1024, depth=24, num_heads=16,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                mae_lr=2.5e-4, regressor_lr=2.5e-4):

        super().__init__()

        self.autoencoder = MaskedAutoencoderViT(img_size=img_size, 
        patch_size=patch_size, 
        in_chans=in_chans, 
        embed_dim=embed_dim, 
        depth=depth, 
        num_heads=num_heads, 
        decoder_embed_dim=decoder_embed_dim, 
        decoder_depth=decoder_depth, 
        decoder_num_heads=decoder_num_heads, 
        mlp_ratio=mlp_ratio, 
        norm_layer=norm_layer
        )    
        self.age_regressor = AgeRegressor(output_dim=1)
        self.automatic_optimization = False


        self.r2 = R2Score()
        self.mae_lr = mae_lr
        self.regressor_lr = regressor_lr

        self.save_hyperparameters()

    def configure_optimizers(self):

        autoencoder_optimizer = AdamW(self.autoencoder.parameters(), lr=self.mae_lr, betas=(0.9, 0.95))
        auto_encoder_scheduler = LinearWarmupCosineAnnealingLR(autoencoder_optimizer, warmup_epochs=40, max_epochs=400)


        regressor_optimizer = AdamW(self.age_regressor.parameters(), lr=self.regressor_lr)
        regressor_scheduler = LinearWarmupCosineAnnealingLR(regressor_optimizer, warmup_epochs=40, max_epochs=400)
        return [autoencoder_optimizer, regressor_optimizer], [auto_encoder_scheduler, regressor_scheduler]
    
    def training_step(self, batch, batch_idx):
        
        eegs, age = batch

        mae_optimizer, _ = self.optimizers()
        reconstruction_loss, *_, latent = self.autoencoder(eegs)

        mae_optimizer.zero_grad() 
        self.manual_backward(reconstruction_loss, retain_graph=True)
        mae_optimizer.step()

        self.log("train_mae_loss", reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.trainer.is_last_batch: 
            mae_scheduler, _ = self.lr_schedulers()
            mae_scheduler.step()
        return reconstruction_loss

    # def training_epoch_end(self):
    #     for lr_scheduler in self.lr_schedulers():
    #         lr_scheduler.step()
    
    
    def evaluate_reconstruction_step(self, target_patch, pred_patch, split="train"):
        
        self.logger.experiment.log({split : wandb.plot.line_series(
          xs=[i for i in range(target_patch.shape[0])],
          ys=[[t+idx for t in ch] for idx, ch in enumerate(target_patch[:6])] \
          + [[t+idx for t in ch] for idx, ch in enumerate(pred_patch[:6])],
          keys=["target_"+str(i) for i in range(target_patch[:6].shape[0])] \
          + ["pred_"+str(i) for i in range(target_patch[:6].shape[0])],
          title="Target vs Predicted Channels",
          xname="Time")})

        pred_patch = pred_patch / torch.linalg.norm(pred_patch, dim=-1, keepdims=True)
        target_patch = target_patch / torch.linalg.norm(target_patch, dim=-1, keepdims=True)
        cos_sim_inter_ch = (target_patch*pred_patch).sum(-1)
        cos_sim_intra_ch = torch.diag(cos_sim_inter_ch).mean()
        self.log(split+"_cosine similarity within channels", cos_sim_intra_ch, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def validation_step(self, batch, batch_idx, dataloader_idx): 
        eegs, age = batch
        
        _, opt_regressor = self.optimizers() 

        if dataloader_idx == 0:

            with torch.no_grad():
                reconstruction_loss, pred, target, mask, latent = self.autoencoder(eegs)
            torch.set_grad_enabled(True)
            
            opt_regressor.zero_grad()
            pred_age = self.age_regressor(latent)
            age_loss = nn.functional.l1_loss(age.squeeze(), pred_age.squeeze())
            self.manual_backward(age_loss)
            opt_regressor.step()

            age_r2  = self.r2(pred_age.squeeze(), age.squeeze())
            if self.trainer.is_last_batch: 
                _, regressor_scheduler = self.lr_schedulers()
                regressor_scheduler.step()
                
        if dataloader_idx == 1:
            
            reconstruction_loss, pred, target, mask, latent = self.autoencoder(eegs)
            pred_age = self.age_regressor(latent)
            age_loss = nn.functional.l1_loss(age.squeeze(), pred_age.squeeze())
            age_r2  = self.r2(pred_age.squeeze(), age.squeeze())
        
        splits = ["train", "val"]
        split = splits[dataloader_idx]
        self.log(split+'_mae_loss', reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(split+'_age_loss', age_loss, prog_bar=True, logger=True)
        self.log(split+'_age_r2', age_r2, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.visualize(mask, target, pred, split)

        return reconstruction_loss

    def visualize_flattened(self, mask, target, pred, split="train"): 
        inverted_mask = 1 - mask
        expanded_mask = inverted_mask[0].unsqueeze(-1)
        masked_target = target[0] * expanded_mask


        show_rate = int((masked_target.reshape(-1).shape[0] * 0.05 ) // 1)

        reshaped_mask = expanded_mask.expand_as(target).float()[:show_rate].cpu()
        flattened_mask = reshaped_mask.reshape(-1)[:show_rate].cpu()
        flattened_masked_target = masked_target.view(-1)[:show_rate].cpu()
        flattened_target = target[0].view(-1)[:show_rate].cpu()
        flattened_pred = pred[0].view(-1)[:show_rate].cpu()

        
        fig, ax = plt.subplots(4, figsize=(15, 5))
        ax[0].plot([i for i in range(flattened_target.shape[-1])], flattened_target.float())
        ax[0].set_title("target")
        ax[1].plot([i for i in range(flattened_masked_target.shape[-1])], flattened_masked_target.float())
        ax[1].set_title("masked target")
        ax[2].plot([i for i in range(flattened_mask.shape[-1])], flattened_mask.float())
        ax[2].set_title("mask")
        ax[3].plot([i for i in range(flattened_pred.shape[-1])], flattened_pred.float().detach().numpy())
        ax[3].set_title("prediction")
        wandb.log({"signals_{}".format(split): fig})

    def visualize(self, mask, target, pred, split="train"): 
        inverted_mask = 1 - mask
        expanded_mask = inverted_mask[0].unsqueeze(-1)
        masked_target = target[0] * expanded_mask
        
        ch_idx = 0
        target_channel = []
        masked_channel = []
        pred_channel = []
        mask_channel = []
        for patch_idx in range(target.shape[1]):
            pred_patch = pred[0, patch_idx, :].view(*self.autoencoder.patch_size)
            target_patch = target[0, patch_idx, :].view(*self.autoencoder.patch_size)
            target_channel.append(target_patch[ch_idx, :])
            pred_channel.append(pred_patch[ch_idx, :])
            masked_channel.append(target_patch[ch_idx, :] * (1-mask[0, patch_idx]))
        target_channel = torch.cat(target_channel)
        pred_channel = torch.cat(pred_channel)
        masked_channel = torch.cat(masked_channel)
        mask_channel = masked_channel == 0
        


      
        fig, ax = plt.subplots(4, figsize=(15, 5))
        ax[0].plot(target_channel.cpu().float()[:1000])
        ax[0].set_title("target")
        ax[1].plot(pred_channel.cpu().float().detach().numpy()[:1000])
        ax[1].set_title("reconstruction")
        ax[2].plot(masked_channel.cpu().float()[:1000])
        ax[2].set_title("masked")
        ax[3].plot(mask_channel.cpu().float()[:1000])
        ax[3].set_title("mask")
        fig.tight_layout()
        wandb.log({"signals_{}".format(split): fig})

