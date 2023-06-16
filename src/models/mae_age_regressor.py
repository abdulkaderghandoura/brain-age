
from mae import MaskedAutoencoderViT
import lightning.pytorch as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch 

from torchmetrics import R2Score
class AgeRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AgeRegressor, self).__init__()

        hidden_size = input_dim // 2 
        self.linear_one = nn.Linear(input_dim, hidden_size)
        self.batch_norm_one = nn.BatchNorm1d(hidden_size)
        self.linear_two = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()


        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        age = self.linear_two(self.relu(self.batch_norm_one(self.linear_one(x))))
        # age = self.linear(x)
        return age


class MAE_AGE(pl.LightningModule):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                embed_dim=1024, depth=24, num_heads=16,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):

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
        mask_ratio = 0.75 
        visible_patches = int(self.autoencoder.num_patches * (1 - mask_ratio)) + 1 
        self.age_regressor = AgeRegressor(visible_patches * embed_dim, 1)
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.r2 = R2Score()

    def configure_optimizers(self):

        autoencoder_optimizer = AdamW(self.autoencoder.parameters(), lr=1e-3, betas=(0.9, 0.95))
        
        auto_encoder_scheduler = LinearWarmupCosineAnnealingLR(autoencoder_optimizer, warmup_epochs=40, max_epochs=400)
        # auto_encoder_scheduler = CosineAnnealingWarmRestarts(autoencoder_optimizer, 400)


        regressor_optimizer = AdamW(self.age_regressor.parameters(), lr=1e-3)
        # regressor_scheduler = CosineAnnealingWarmRestarts(regressor_optimizer, 400)
        regressor_scheduler = LinearWarmupCosineAnnealingLR(regressor_optimizer, warmup_epochs=40, max_epochs=400)
        return [autoencoder_optimizer, regressor_optimizer], [auto_encoder_scheduler, regressor_scheduler]

    def training_step(self, batch, batch_idx):
        eegs, age = batch

        mae_optimizer, age_optimizer = self.optimizers() 
        
        reconstruction_loss, _, _, latent = self.autoencoder(eegs)
        # embedding = torch.squeeze(latent, dim=1)
        # self.print(latent.size())


        mae_optimizer.zero_grad() 
        self.manual_backward(reconstruction_loss, retain_graph=True)
        mae_optimizer.step()

        age_prediction = self.age_regressor(latent).squeeze()
        
        age_loss = nn.functional.mse_loss(age_prediction.float(), age.float())
        
        self.autoencoder.freeze()
        age_optimizer.zero_grad()
        self.manual_backward(age_loss)
        age_optimizer.step()
        self.autoencoder.unfreeze()


        self.log("train_mae_loss", reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_age_loss", age_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return reconstruction_loss

    def on_train_epoch_end(self):
        for lr_scheduler in self.lr_schedulers():
            lr_scheduler.step()

    def validation_step(self, batch, batch_idx):

        eegs, age = batch

        mae_optimizer, age_optimizer = self.optimizers() 

        reconstruction_loss, _, _, latent = self.autoencoder(eegs)

        age_prediction = self.age_regressor(latent).squeeze()
        
        age_loss = nn.functional.mse_loss(age_prediction, age)
        age_r2  = self.r2(age_prediction, age)
        self.log("val_mae_loss", reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_age_loss', age_loss, prog_bar=True, logger=True)
        self.log('val_age_r2', age_r2, prog_bar=True, logger=True)
        return reconstruction_loss