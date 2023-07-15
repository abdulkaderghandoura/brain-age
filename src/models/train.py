import argparse
import numpy as np
import torch 
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import lightning.pytorch as pl
import wandb
from functools import partial
import os
import sys
sys.path.append('../utils/')
from transforms import channelwide_norm, channelwise_norm, _clamp, _randomcrop, _compose
from mae_age_regressor import MAE_AGE
from dataset import EEGDataset
from mae import MaskedAutoencoderViT

def get_args_parser():

    parser = argparse.ArgumentParser('MAE pretraining', add_help=False)
    parser.add_argument('--experiment_name', default='pretraining_mae')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=400, type=int, 
                        help='total number of epochs') 
    # model parameters 
    parser.add_argument('--mae_age', default=False, type=bool, 
                        help='run mae with age regression head or not')
    parser.add_argument('--patch_size_one', default=63, type=int,
                         help='patch size for the channel space')
    parser.add_argument('--patch_size_two', default=100, type=int,
                        help='patch size for the time')
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
    parser.add_argument('--lr_mae', default=2.5e-4, type=float, 
                        help='learning rate to train the masked autoencoder with')    
    parser.add_argument('--lr_regressor', default=2.5e-4, type=float, 
                        help='learning rate to train the regression head with')
    parser.add_argument('--pixel_norm', default=False, type=bool, 
                        help='normalize the output pixels before computing the loss')
    # dataset parameters 
    parser.add_argument('--mae_train_dataset', default=['bap'], type=list, nargs='+', 
                        help='dataset for training mae eg. bap, hbn, lemon')
    parser.add_argument('--mae_val_dataset', default=['bap'], type=list, nargs='+', 
                        help='dataset for validating mae eg. bap, hbn, lemon')
    parser.add_argument('--regressor_train_dataset', default=['bap'], type=list, nargs='+', 
                        help='dataset for training the regressor, only effectice if mae_age is true\
                        eg. bap, hbn, lemon')
    parser.add_argument('--oversample', default=False, type=bool, 
                        help='to oversample bap when training on target and external dataset only effective \
                        if multiple dataset is specified in mae_train_dataset')
    parser.add_argument('--num_channels', default=63, type=int,
                        help='number of channels in the dataset eg. 63 for bap and hbn and 61 for lemon')
    parser.add_argument('--input_time', default=10, type=int,
                        help='number of seconds in the input')
    parser.add_argument('--fs', default=100, type=int,
                        help='data sampling frequency')      

    # transforms 
    parser.add_argument('--standardization', default='channelwise', type=str,
                       help='standardization applied to the model input, e.g. channelwise, channelwide')
    parser.add_argument('--clamp_val', default=4, type=float, 
                        help='the input to the model will be limited between (-clamp_val, clamp_val)')
    
    # callbacks
    parser.add_argument('--checkpoint_callback', default=False, type=bool, 
                        help='to save best and last checkpoints')
    parser.add_argument('--early_stopping', default=False, type=bool, 
                        help='to early stop the training when validation loss is stable ')
    
    # other parameters 
    parser.add_argument('--device', default='0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int, 
                        help='number of workers for the dataloaders')    
    parser.add_argument('--overfit_batches', default=1.0, type=float, 
                        help='to debug model on a fraction of batches')

    return parser
    

def main(args):

    # To properly utilize 'NVIDIA A40' For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision('medium')

    # Setting seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # number of samples in the input 
    num_of_samples = args.input_time * args.fs

    # w/o online linear probing 
    if args.mae_age:
        model = MAE_AGE(EEG_size=(args.num_channels, num_of_samples), \
                                            patch_size=(args.patch_size_one, args.patch_size_two), \
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
        model = MaskedAutoencoderViT(EEG_size=(args.num_channels, num_of_samples), \
                                    patch_size=(args.patch_size_one, args.patch_size_two), \
                                    embed_dim=args.embed_dim, 
                                    depth=args.depth, 
                                    num_heads=args.num_heads, 
                                    decoder_embed_dim=args.decoder_embed_dim, 
                                    decoder_depth=args.decoder_depth, 
                                    decoder_num_heads=args.decoder_num_heads,
                                    mlp_ratio=args.mlp_ratio, 
                                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                    norm_pix_loss=args.pixel_norm,
                                    lr=args.lr_mae,
                                    )

    # initializing and composing the transforms 
    if args.standardization == "channelwise":
        norm = channelwise_norm
    elif args.standardization == "channelwide":
        norm = channelwide_norm 
    
    randomcrop = partial(_randomcrop, seq_len=num_of_samples)
    clamp = partial(_clamp, dev_val=args.clamp_val)
    composed_transforms = partial(_compose, transforms=[
                                                        randomcrop, 
                                                        norm, 
                                                        clamp
                                                        ])
    # Initializing training and validation dataloaders 
    autoencoder_train_dataset = EEGDataset(args.mae_train_dataset, ['train'], transforms=composed_transforms, oversample=args.oversample)
    autoencoder_train_dataloader = DataLoader(autoencoder_train_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                pin_memory=True, 
                                shuffle=True)
    if args.mae_age:
        regressor_train_dataset = EEGDataset(args.regressor_train_dataset, ['train'], transforms=composed_transforms)
        regressor_train_dataloader = DataLoader(regressor_train_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers, 
                                    pin_memory=True, 
                                    shuffle=True)

    val_dataset = EEGDataset(args.mae_val_dataset, ['val'], transforms=composed_transforms)
    validation_dataloader =  DataLoader(val_dataset, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers, 
                                        pin_memory=True, 
                                        )
    # starting wandb logger 
    wandb.login()
    logger = pl.loggers.WandbLogger(project="brain-age", name=args.experiment_name, 
                                    save_dir="wandb/", 
                                    log_model=True # logs the last model 
                                    )
    # lr monitor callback to log the lr 
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]

    # early stopping callback to stop training when model converges 
    if args.early_stopping:
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-7, patience=10, verbose=False, mode="min")
        callbacks.append(early_stop_callback)
    
    # checkpoint callback to save last and best checkpoints localy 
    if args.checkpoint_callback:
        checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',  # Metric to monitor for saving the best model
                filename='best_model',  # Filename pattern for saved models
                save_top_k=1,  # Number of best models to save (set to 1 for the best model only)
                mode='min',  # Mode of the monitored metric (minimize val_loss in this case)
                dirpath='../../models/checkpoints/{}'.format(args.experiment_name),
                save_last=True
            )
        callbacks.append(checkpoint_callback)

    
    trainer = pl.Trainer(
                        overfit_batches=args.overfit_batches, # to debug on smaller batches 
                        deterministic=True, # to ensure reproducibility 
                        devices=[0], 
                        callbacks=callbacks, 
                        check_val_every_n_epoch=1, 
                        max_epochs=args.epochs, 
                        accelerator="gpu", 
                        logger=logger,
                        precision="bf16-mixed", 
                        )

    if args.mae_age: 
        # if onlinea linear probing, regressor head is trained in the validation
        # 2 dataloaders are sent one for training the regressor head 
        # and the second for validating mae and regressor head 
        val_data = [regressor_train_dataloader, validation_dataloader]
    else: 
        val_data = validation_dataloader
    
    # start training 
    trainer.fit(
        model=model, 
        train_dataloaders=autoencoder_train_dataloader, 
        val_dataloaders=val_data
        )
    # to mark a run as finished, and finish uploading all data.
    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)