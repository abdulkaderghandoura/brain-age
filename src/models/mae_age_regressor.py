import wandb
from mae import MaskedAutoencoderViT
import lightning.pytorch as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch.nn as nn
from torch.optim import Adam, AdamW, LBFGS
import torch
from train_utils import visualize
from matplotlib import pyplot as plt
from torchmetrics import R2Score
    
class AgeRegressor(nn.Module):
    "The linear head used for linear probing, finetuning"
    def __init__(self, input_dim, output_dim):
        """
        Initializes the AgeRegressor model.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output predictions.
        """
        super(AgeRegressor, self).__init__()
        
        # model layers 
        self.norm = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the AgeRegressor model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        if len(x.shape) > 2:
            x = x[:, 0]
        x = self.norm(x) # Apply batch normalization
        x = self.linear(x) # Apply linear transformation
        return x

class MAE_AGE(pl.LightningModule):
    def __init__(self, EEG_size=(63, 1000), patch_size=(1, 100), in_chans=1,
                embed_dim=384, depth=3, num_heads=6,
                decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                lr_mae=1e-4, lr_regressor=2.5e-4):
                """
        Initialize the MAE_AGE model.

        Args:
            EEG_size (tuple): Size of the input EEG signals (channels, time steps).
            patch_size (tuple): Size of the input patches (channels, time steps).
            in_chans (int): Number of input channels.
            embed_dim (int): Dimensionality of the embeddings.
            depth (int): Depth of the model.
            num_heads (int): Number of attention heads.
            decoder_embed_dim (int): Dimensionality of the decoder embeddings.
            decoder_depth (int): Depth of the decoder.
            decoder_num_heads (int): Number of attention heads in the decoder.
            mlp_ratio (float): Ratio of the hidden dimension to the embedding dimension in the MLP.
            norm_layer (torch.nn.Module): Normalization layer.
            norm_pix_loss (bool): Whether to normalize the pixel loss.
            lr_mae (float): Learning rate for the MAE optimizer.
            lr_regressor (float): Learning rate for the regressor optimizer.
        """
        super().__init__()
        # intializing the masked autoencoder 
        self.autoencoder = MaskedAutoencoderViT(EEG_size=EEG_size, 
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
        # initializing the regressor head 
        self.age_regressor = AgeRegressor(input_dim=embed_dim, output_dim=1)
        # turning off pytorch lightning automatic optimization 
        self.automatic_optimization = False

        self.r2 = R2Score()
        self.mae_lr = lr_mae
        self.regressor_lr = lr_regressor
        self.save_hyperparameters() # useful to log hyperparameters into wandb 


    def configure_optimizers(self):
        """
        Configure optimizer and scheduler masked autoencoder and the regressor head 

        Returns:
            List[torch.optim.Optimizer]: Mae and regressor optimizers
            List[torch.optim.lr_scheduler._LRScheduler]: Mae and regressor schedulers
        """

        autoencoder_optimizer = AdamW(self.autoencoder.parameters(), lr=self.mae_lr, betas=(0.9, 0.95))
        auto_encoder_scheduler = LinearWarmupCosineAnnealingLR(autoencoder_optimizer, warmup_epochs=10, max_epochs=100)


        regressor_optimizer = AdamW(self.age_regressor.parameters(), lr=self.regressor_lr)
        regressor_scheduler = LinearWarmupCosineAnnealingLR(regressor_optimizer, warmup_epochs=40, max_epochs=400)
        
        return [autoencoder_optimizer, regressor_optimizer], [auto_encoder_scheduler, regressor_scheduler]
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step on a batch of data.

        Args:
            batch: The batch of data from the dataloader.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the training step.
        """
        
        eegs, age = batch

        # masked autoencoder optimizer 
        mae_optimizer, _ = self.optimizers()

        mae_optimizer.zero_grad() 
        # computing fwd path for mae and reconstruction loss 
        reconstruction_loss, *_, latent = self.autoencoder(eegs)
        # backpropagating mae using the reconstruction loss 
        self.manual_backward(reconstruction_loss, retain_graph=True)
        # mae optimizer step 
        mae_optimizer.step()
        # logging the reconstruction loss 
        self.log("train_mae_loss", reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # if it is the last batch in this epoch, perform one step for the learning rate scheduler 
        if self.trainer.is_last_batch: 
            mae_scheduler, _ = self.lr_schedulers()
            mae_scheduler.step()
        return reconstruction_loss
    
    def evaluate_reconstruction_step(self, target_patch, pred_patch, split="train"):
        """
        logs the cosine similarity.

        Args:
            target_patch (torch.Tensor): Target patch for comparison.
            pred_patch (torch.Tensor): Predicted patch for comparison.
            split (str): Split identifier (e.g., 'train', 'val').

        """

        pred_patch = pred_patch / torch.linalg.norm(pred_patch, dim=-1, keepdims=True)
        target_patch = target_patch / torch.linalg.norm(target_patch, dim=-1, keepdims=True)
        cos_sim_inter_ch = (target_patch*pred_patch).sum(-1)
        cos_sim_intra_ch = torch.diag(cos_sim_inter_ch).mean()
        self.log(split+"_cosine similarity within channels", cos_sim_intra_ch, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        torch.cuda.empty_cache()


    def validation_step(self, batch, batch_idx, dataloader_idx): 
        """
        Perform a training step for the regressor using 
        dataloader_idx == 0 
        and validation step for mae using dataloader_idx ==1 
        on a batch of data.

        Args:
            batch: The batch of data from the dataloader.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the current dataloader.

        Returns:
            torch.Tensor: The loss value for the validation step.
        """
        eegs, age = batch
        # regressor head optimizer 
        _, opt_regressor = self.optimizers() 

        if dataloader_idx == 0: # dataloader used to train the regressor 

            with torch.no_grad():
                # masked autoencoder fwd path 
                reconstruction_loss, pred, target, mask, latent = self.autoencoder(eegs, set_masking_seed=False)
            torch.set_grad_enabled(True)
            
            opt_regressor.zero_grad()
            # predicting age 
            pred_age = self.age_regressor(latent)
            # computing the loss 
            age_loss = nn.functional.l1_loss(pred_age.squeeze(), age.squeeze())
            # backpropagation using the loss 
            self.manual_backward(age_loss)
            # regressor optimizer step 
            opt_regressor.step()
            # computing the r2 for the age 
            age_r2  = self.r2(pred_age.squeeze(), age.squeeze())

                
        if dataloader_idx == 1: # dataloader used to validate the age regressor and masked autoencode r
            
            # masked autoencoder fwd path 
            reconstruction_loss, pred, target, mask, latent = self.autoencoder(eegs, set_masking_seed=False)
            # regressor head fwd path 
            pred_age = self.age_regressor(latent)
            # computing the loss and r2 for the age 
            age_loss = nn.functional.l1_loss(pred_age.squeeze(), age.squeeze())
            age_r2  = self.r2(pred_age.squeeze(), age.squeeze())
        
        # logging reconstruction, age loss and age r2 
        splits = ["train", "val"]
        split = splits[dataloader_idx]
        self.log(split+'_mae_loss', reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(split+'_age_loss', age_loss, prog_bar=True, logger=True)
        self.log(split+'_age_r2', age_r2, prog_bar=True, logger=True)

        # if it is the last batch and the dataloader training age regressor 
        # perfom one scheduler step 
        if self.trainer.is_last_batch and dataloader_idx == 0: 
            _, regressor_scheduler = self.lr_schedulers()
            regressor_scheduler.step()
        
        if batch_idx == 0:
            # visualize the reconstruction 
            visualize(self.EEG_size[1], mask, target, pred, split)

        return reconstruction_loss
