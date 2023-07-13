import argparse
from torch.utils.data import DataLoader
import torch 
import lightning.pytorch as pl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import wandb
from functools import partial
import sys
sys.path.append('../utils/')
from transforms import channelwide_norm, channelwise_norm, _clamp, _randomcrop, _compose
from dataset import EEGDataset
from mae import MaskedAutoencoderViT

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import numpy as np
# torch.cuda.empty_cache() 
from mae_age_regressor import MAE_AGE
def get_args_parser():
    parser = argparse.ArgumentParser('MAE training', add_help=False)
    parser.add_argument('--experiment_name', default='mae_batch_training_EEG_AdamW_optim')
    
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=400, type=int)


    parser.add_argument('--standardization', default='channelwise', type=str,
                       help='standardization applied to the model input, e.g. channelwise, channelwide')
    
    parser.add_argument('--crop_len', default=1000, type=int,
                       help='# of time samples of the random crop applied to the model input')
    
    parser.add_argument('--clamp_val', default=4, type=float, 
                        help='the input to the model will be limited between (-clamp_val, clamp_val)')
    # data parameters 

    parser.add_argument('--regressor_train_dataset', default=['lemon'], type=list, nargs='+', 
                        help='dataset for training eg. bap, hbn, lemon')
    
    parser.add_argument('--mae_train_dataset', default=['lemon'], type=list, nargs='+', 
                        help='dataset for training eg. bap, hbn, lemon')
    
    parser.add_argument('--mae_val_dataset', default=['lemon'], type=list, nargs='+', 
                        help='dataset for training eg. bap, hbn, lemon')

    parser.add_argument('--num_channels', default=63, type=int,
                        help='number of channels in the input')
    parser.add_argument('--input_time', default=10, type=int,
                        help='number of seconds in the input')
    parser.add_argument('--fc', default=100, type=int,
                        help='data sampling frequency')       
    # model parameters 

    parser.add_argument('--patch_size_one', default=1, type=int,
                         help='patch size for the space')

    parser.add_argument('--patch_size_two', default=100, type=int,
                        help='patch size for the time')
    
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)


    # to avoid a bottleneck 256 which is the number of cpus on the machine
    parser.add_argument('--num_workers', default=10, type=int, 
                        help='number of workers for the dataloaders')
    
    parser.add_argument('--embed_dim', default=384, type=int, 
                        help='embedding dimension of the encoder')
    parser.add_argument('--depth', default=2, type=int, 
                        help='the number of blocks of the encoder')
    parser.add_argument('--num_heads', default=4, type=int, 
                        help='the number of attention heads of the encoder')
    parser.add_argument('--decoder_embed_dim', default=256, type=int, 
                        help='the embedding dimension of the decoder')
    parser.add_argument('--decoder_depth', default=2, type=int, 
                        help='the number of blocks of the decoder')
    parser.add_argument('--decoder_num_heads', default=8, type=int, 
                    help='number of attention heads of the decoder')
    parser.add_argument('--mlp_ratio', default=4., type=float, 
                        help='ratio of mlp hidden dim to embedding dim')
    
    
    parser.add_argument('--mae_age', default=False, type=bool, 
                        help='run mae with age regression head or not')
    parser.add_argument('--oversample', default=False, type=bool, 
                        help='to oversample the minority dataset when training on target and external dataset ')
    parser.add_argument('--overfit_batches', default=1.0, type=float, 
                        help='to debug model on a fraction of batches')
    
    # callbacks
    parser.add_argument('--checkpoint_callback', default=False, type=bool, 
                        help='to save best and last checkpoints')
    parser.add_argument('--early_stopping', default=False, type=bool, 
                        help='to early stop the training when validation loss is stable ')
    
    parser.add_argument('--lr_mae', default=2.5e-4, type=float, 
                        help='learning rate to train the masked autoencoder with')    
    parser.add_argument('--lr_regressor', default=2.5e-4, type=float, 
                        help='learning rate to train the regression head with')
    
    parser.add_argument('--pixel_norm', default=False, type=bool, 
                        help='normalize the output pixels before computing the loss')


    return parser
    

def main(args):

    # using a CUDA device ('NVIDIA A40') that has Tensor Cores. 
    # To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
    # which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    
    torch.set_float32_matmul_precision('medium')

    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    #size of the input = # of seconds * sampling frequency 



    if args.mae_age:
        model = MAE_AGE(img_size=(63, args.input_time * 135), \
                                            patch_size=(21, 90), \
                                            in_chans=1, 
                                            embed_dim=args.embed_dim, 
                                            depth=args.depth, 
                                            num_heads=args.num_heads, 
                                            decoder_embed_dim=args.decoder_embed_dim, 
                                            decoder_depth=args.decoder_depth, 
                                            decoder_num_heads=args.decoder_num_heads,
                                            mlp_ratio=args.mlp_ratio, 
                                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                            norm_pix_loss=args.pixel_norm,
                                            lr_mae=args.lr_mae,
                                            lr_regressor=args.lr_regressor
                                            )
    else: 
        model = MaskedAutoencoderViT(img_size=(args.num_channels, args.input_time * args.fc), \
                                    patch_size=(args.patch_size_one, args.patch_size_two), \
                                    in_chans=1, 
                                    embed_dim=args.embed_dim, 
                                    depth=args.depth, 
                                    num_heads=args.num_heads, 
                                    decoder_embed_dim=args.decoder_embed_dim, 
                                    decoder_depth=args.decoder_depth, 
                                    decoder_num_heads=args.decoder_num_heads,
                                    mlp_ratio=args.mlp_ratio, 
                                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                    norm_pix_loss=args.pixel_norm,
                                    lr_mae=args.lr_mae,
                                    lr_regressor=args.lr_regressor
                                    )

    
    if args.standardization == "channelwise":
        norm = channelwise_norm
    elif args.standardization == "channelwide":
        norm = channelwide_norm 
    randomcrop = partial(_randomcrop, seq_len=args.crop_len)
    clamp = partial(_clamp, dev_val=args.clamp_val)
    composed_transforms = partial(_compose, transforms=[
                                                        randomcrop, 
                                                        norm, 
                                                        clamp
                                                        ])



    # train_dataset = EEGDataset(args.train_dataset, ['train'], transforms=composed_transforms, oversample=args.oversample)
    autoencoder_train_dataset = EEGDataset(['bap'], ['train'], transforms=composed_transforms, oversample=True)
    autoencoder_train_dataloader = DataLoader(autoencoder_train_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                pin_memory=True, 
                                shuffle=True)
    regressor_train_dataset = EEGDataset(args.regressor_train_dataset, ['train'], transforms=composed_transforms, oversample=False)
    regressor_train_dataloader = DataLoader(regressor_train_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                pin_memory=True, 
                                shuffle=True)

    val_dataset = EEGDataset(['bap'], ['val'], transforms=composed_transforms)
    validation_dataloader =  DataLoader(val_dataset, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers, 
                                        pin_memory=True, 
                                        shuffle=True
                                        )


    wandb.login()
    logger = pl.loggers.WandbLogger(project="brain-age", name=args.experiment_name, 
                                    save_dir="wandb/", log_model=True)
    
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-7, patience=3, verbose=False, mode="min")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Metric to monitor for saving the best model
        filename='best_model',  # Filename pattern for saved models
        save_top_k=1,  # Number of best models to save (set to 1 for the best model only)
        mode='min',  # Mode of the monitored metric (minimize val_loss in this case)
        dirpath='../../models/checkpoints/{}'.format(args.experiment_name),
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [lr_monitor]

    if args.early_stopping:
        callbacks.append(early_stop_callback)
    if args.checkpoint_callback:
        callbacks.append(checkpoint_callback)

    
    trainer = pl.Trainer(
                        overfit_batches=args.overfit_batches,
                        deterministic=True, # to ensure reproducibility 
                        devices=[0], 
                        callbacks=callbacks, 
                        check_val_every_n_epoch=1,
                        max_epochs=args.epochs, 
                        accelerator="gpu", 
                        logger=logger,
                        precision="bf16-mixed", 
                        # fast_dev_run=True, 
                        )

    if args.mae_age: 
        val_data = [regressor_train_dataloader, validation_dataloader]
    else: 
        val_data = validation_dataloader
    trainer.fit(
        model=model, 
        train_dataloaders=autoencoder_train_dataloader, 
        val_dataloaders=val_data
        )
    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)