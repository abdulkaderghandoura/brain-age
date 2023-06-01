import argparse
from torch.utils.data import DataLoader
import torch 
import lightning.pytorch as pl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import wandb

from dataset import EEGDataset
from mae import MaskedAutoencoderViT

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# torch.cuda.empty_cache() 

def get_args_parser():
    parser = argparse.ArgumentParser('MAE training', add_help=False)
    parser.add_argument('--experiment_name', default='mae_batch_training_EEG_AdamW_optim')
    
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=400, type=int)

    parser.add_argument('--dataset', default='bap', type=str, 
                        help='dataset for training eg. bap, hbn, lemon')
    
    parser.add_argument('--standardization', default='channelwise', type=str,
                       help='standardization applied to the model input, e.g. channelwise, channelwide')
    
    parser.add_argument('--crop_len', default=4050, type=int,
                       help='# of time samples of the random crop applied to the model input')
    
    parser.add_argument('--clamp_val', default=20, type=float, 
                        help='the input to the model will be limited between (-clamp_val, clamp_val)')
    
    # model parameters 
    parser.add_argument('--input_time', default=30, type=int,
                        help='number of seconds in the input')

    parser.add_argument('--patch_size', default=90, type=int,
                        help='patch input size')
    
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # to avoid a bottleneck 256 which is the number of cpus on the machine
    parser.add_argument('--num_workers', default=256, type=int, 
                        help='number of workers for the dataloaders')
    #logging
    

    return parser
    

def main(args):

    # using a CUDA device ('NVIDIA A40') that has Tensor Cores. 
    # To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
    # which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    
    torch.set_float32_matmul_precision('medium')

    seed = args.seed 
    torch.manual_seed(seed)
    #size of the input = # of seconds * sampling frequency 
    model = MaskedAutoencoderViT(img_size=(65, args.input_time * 135), \
                                        patch_size=(65, args.patch_size), \
                                        in_chans=1)
    
    import sys
    sys.path.append('~/brain-age/src/utils')
    from transforms import channelwide_norm, channelwise_norm, _clamp, _randomcrop
    if args.standardization == "channelwise":
        norm = channelwise_norm
    elif args.standardization == "channelwide":
        norm = channelwide_norm 
    randomcrop = partial(_randomcrop, seq_len=args.crop_len)
    clamp = partial(_clamp, dev_val=args.clamp_val)
    transforms = partial(_compose, transforms=[randomcrop, norm, clamp])

    train_dataset = EEGDataset(args.dataset, 'train', transforms=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    val_dataset = EEGDataset(args.dataset, 'val', transforms=transforms)
    validation_dataloader =  DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)


    # wandb.login()
    logger = pl.loggers.WandbLogger(project="brain-age", name=args.experiment_name, 
                                    save_dir="/wandb/", log_model=False)
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=25, verbose=False, mode="max")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Metric to monitor for saving the best model
        filename='best_model_{epoch:02d}-{val_loss:.4f}',  # Filename pattern for saved models
        save_top_k=1,  # Number of best models to save (set to 1 for the best model only)
        mode='min',  # Mode of the monitored metric (minimize val_loss in this case)
        dirpath='../../models/checkpoints/{}'.format(args.experiment_name),
        save_last=True
    )

    trainer = pl.Trainer(
                        # overfit_batches=1.0,
                        callbacks=[checkpoint_callback, early_stop_callback], 
                        max_epochs=args.epochs, 
                        accelerator="gpu", 
                        logger=logger,
                        precision="bf16-mixed"
                        )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)