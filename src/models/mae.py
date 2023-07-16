# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# References:
# original implementation: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# -----------------------------------------------------------------------------
# Changes made to the original implementation: 
#     - adapted implementation to work with pytorch lightning 
#     - adapted implementation to work on non square input EEG instead of squared imgs 
#     - added an option for fixed masking across batches
#     - added useful loggings 
# ------------------------------------------------------------------------------
from timm.models.vision_transformer import PatchEmbed, Block
import lightning.pytorch as pl
import matplotlib.pyplot as plt 
import wandb 
import torch.nn as nn
import torch 
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pos_embed import get_2d_sincos_pos_embed
from train_utils import visualize
class MaskedAutoencoderViT(pl.LightningModule):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, EEG_size=(63, 1000), patch_size=(1, 100), in_chans=1,
                 embed_dim=384, depth=2, num_heads=4,
                 decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, lr=2.5e-4):
        """
        initializes the masked autoencoder with a Vit backbone

        Args:
            EEG_size (tuple, optional): size of the input EEG. Defaults to (63, 1000).
            patch_size (tuple, optional): size of the patches. Defaults to (1, 100).
            in_chans (int, optional): number of input channels, relevant when working on imgs. Defaults to 1.
            embed_dim (int, optional): dimension of encoder latent space per patch. Defaults to 384.
            depth (int, optional): number of layer in the encoder transformer. Defaults to 2.
            num_heads (int, optional): number of attention heads for each attention layer in the Transformer encoder.
                                        . Defaults to 4.
            decoder_embed_dim (int, optional): dimension of decoder embedding. Defaults to 256.
            decoder_depth (int, optional): number of layer in the decoder transformer. Defaults to 2.
            decoder_num_heads (int, optional): number of attention heads for each attention layer in the Transformer decoder
                                            . Defaults to 8.
            mlp_ratio (float, optional): Expansion factor of the feed-forward network. Defaults to 4..
            norm_layer (nn.LayerNorm, optional): normal layer. Defaults to nn.LayerNorm.
            norm_pix_loss (bool, optional): to normalize the target. Defaults to False.
            lr (float, optional): base lr. Defaults to 2.5e-4.
        """
        super().__init__()
        
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.grid_size = (self.EEG_size[0]//self.patch_size[0], self.EEG_size[1]//self.patch_size[1])
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(EEG_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0]*patch_size[1]* in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.lr = lr
        self.initialize_weights()
        self.save_hyperparameters()

    def initialize_weights(self):
        """
        performs the following initialization steps:
            - Initializes (and freezes) the pos_embed using sin-cos embedding.
            - Initializes the decoder_pos_embed using sin-cos embedding.
            - Initializes the patch_embed weights using xavier_uniform_ initialization.
            - Initializes self.cls_token and self.mask_token using normal_ initialization.
            - Initializes nn.Linear and nn.LayerNorm.
        """
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size, cls_token=True)
        temp = torch.from_numpy(pos_embed).float().unsqueeze(0)
        self.pos_embed.data.copy_(temp)

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size,  cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initializes the weights of the given module.

        Args:
            m (nn.Module):  The module whose weights need to be initialized.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, EEGs):
        """
        Patchifies the input EEGs tensor into smaller patches.

        Args:
            EEGs (torch.Tensor): Input EEGs tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Patchified tensor of shape (batch_size, num_patches, patch_dim).

        Raises:
            AssertionError: If the height or width of the input EEGs tensor is not divisible by the patch size.

        """
        p1, p2 = self.patch_embed.patch_size
        
        assert EEGs.shape[2] % p1 == 0 and EEGs.shape[3] % p2 == 0
    
        h = EEGs.shape[2] // p1 # number of patches vertically 
        w = EEGs.shape[3] // p2 # number of patches horizontally  

        x = EEGs.reshape(shape=(EEGs.shape[0], EEGs.shape[1], h, p1, w, p2))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(EEGs.shape[0], h * w, p1*p2 * EEGs.shape[1]))
        return x

    def random_masking(self, x, mask_ratio, set_masking_seed=False):
        """
        Perform per-sample random masking by per-sample shuffling.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (N, L, D), where N is the batch size,
                L is the length of the sequence, and D is the dimension of each element.
            mask_ratio (float): Ratio of the sequence length to mask. Should be a value between 0 and 1.
            set_masking_seed (bool, optional): Flag indicating whether to set a fixed seed for the masking noise.
                Defaults to False.

        Returns:
            x_masked (torch.Tensor): Masked sequence tensor of shape (N, len_keep, D), where len_keep
                is the length of the sequence after masking.
            mask (torch.Tensor): Binary mask tensor of shape (N, L), where 0 represents elements to keep
                and 1 represents elements to remove.
            ids_restore (torch.Tensor): Tensor of indices used to restore the original order after shuffling.

        """

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        if set_masking_seed: 
            noise = torch.rand(N, L, generator=torch.Generator(0), device=x.device)
        else:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, set_masking_seed=False):
        """
        Performs forward pass for the encoder

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            mask_ratio (float): Ratio of the sequence length to mask. Should be a value between 0 and 1.
            set_masking_seed (bool, optional): Flag indicating whether to set a fixed seed for the masking noise.
                Defaults to False.

        Returns:
            x (torch.Tensor): Encoded tensor after applying the encoder blocks, of shape (batch_size, num_of_patches, hidden_dim).
            mask (torch.Tensor): Binary mask tensor of shape (batch_size, num_of_patches), where 0 represents elements to keep
                and 1 represents elements to remove.
            ids_restore (torch.Tensor): Tensor of indices used to restore the original order after shuffling.
        """

        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, set_masking_seed)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Performs forward pass for the decoder 

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_of_patches, hidden_dim).
            ids_restore (torch.Tensor): Tensor of indices used to restore the original order after shuffling.

        Returns:
            torch.Tensor: Decoded tensor after applying the decoder blocks, of shape (batch_size, num_of_patches, hidden_dim).

        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, EEGs, pred, mask):
        """
        Compute the forward loss based on the input EEG, predicted patches, and mask.

        Args:
            EEGs (torch.Tensor): Input EEG tensor of shape (N, 1, H, W).
            pred (torch.Tensor): Predicted patch tensor of shape (N, L, p1*p2).
            mask (torch.Tensor): Binary mask tensor of shape (N, L), where 0 represents elements to keep
                and 1 represents elements to remove.

        Returns:
            torch.Tensor: Computed forward loss.

        """
        target = self.patchify(EEGs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward(self, EEGs, mask_ratio=0.75, set_masking_seed=False):
        """
        Performs forward pass 

        Args:
            EEGs (torch.Tensor): Input EGG tensor of shape (N, 1, H, W).
            mask_ratio (float, optional): Ratio of the patches to mask during encoder forward pass.
                Defaults to 0.75.
            set_masking_seed (bool, optional): Flag indicating whether to set a fixed seed for the masking noise
                during encoder forward pass. Defaults to False.

        Returns:
            loss (torch.Tensor): Computed forward loss.
            pred (torch.Tensor): Predicted patch tensor of shape (N, L, p1*p2).
            target (torch.Tensor): Patchified target tensor of shape (N, L, p1*p2).
            mask (torch.Tensor): Binary mask tensor of shape (N, L), where 0 represents elements to keep
            and 1 represents elements to remove.
            latent (torch.Tensor): Encoded latent tensor.

        """
        latent, mask, ids_restore = self.forward_encoder(EEGs, mask_ratio, set_masking_seed)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p1*p2]
        loss = self.forward_loss(EEGs, pred, mask)
        target = self.patchify(EEGs)
        return loss, pred, target, mask, latent

    def training_step(self, batch, batch_idx):
        """
        Performs a training step on a batch of data.

        Args:
            batch: A batch of data, typically containing input EEGs and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The reconstruction loss computed during the training step.

        """
        eegs, _ = batch
        reconstruction_loss, pred, target, mask, latent = self(eegs)
        self.log("train_loss", reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx == 0:
            visualize(self.EEG_size[1], mask, target, pred, 'train')
        return reconstruction_loss
    
    def validation_step(self, batch, batch_idx): 
        """
        Performs a validation step on a batch of data.

        Args:
            batch: A batch of data, typically containing input EEGs and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The reconstruction loss computed during the training step.

        """
        eegs, _= batch
        reconstruction_loss, pred, target, mask, latent = self(eegs)
        self.log("val_loss", reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx == 0:
            visualize(self.EEG_size[1], mask, target, pred, 'val')
        return reconstruction_loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        Returns:
            optimizers (List[torch.optim.Optimizer]): List of optimizers used for training.
            schedulers (List[torch.optim.lr_scheduler._LRScheduler]): List of learning rate schedulers used for training.

        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=40, max_epochs=400)
        return [optimizer], [scheduler]
