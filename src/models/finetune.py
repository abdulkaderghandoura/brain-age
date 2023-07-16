import argparse
import wandb
from functools import partial
import torch 
import numpy as np
import os
import pathlib
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import sys
sys.path.append('../utils/')
import copy
from transforms import channelwide_norm, channelwise_norm, _clamp, _randomcrop, _compose, \
                        amplitude_flip, channels_dropout, time_masking, gaussian_noise
from mae_finetuner import VisionTransformer
from mae import MaskedAutoencoderViT
from dataset import EEGDataset

def get_args_parser():

    parser = argparse.ArgumentParser('MAE finetuning', add_help=False)

    # finetuning parameters
    parser.add_argument('--experiment_name', default='downstream_task')
    parser.add_argument('--batch_size', default=128, type=int,
                            help='Batch size')
    parser.add_argument('--epochs', default=[100, 150], type=int, nargs='+', 
                            help='Maximum number of epochs for linear probing and finetuning, respectively')
    parser.add_argument('--artifact_id', default='h4vidwgf:v210', type=str, 
                            help='name and version of the model artifact to be finetuned')
    parser.add_argument('--lr', default=[1e-2, 1e-4], type=float, nargs='+', 
                            help='learning rates for linear probing and finetuning, respectively')
    parser.add_argument('--mode', default=["linear_probe", "finetune_encoder"] , type=str, nargs='+', help=('Select mode for fine tuning. Can be one of: ',
                        '[linear_probe], [linear_probe, finetune_encoder], [linear_probe, finetune_final_layer], [random_initialization]'))
    parser.add_argument('--augment_data', default=False, type=bool, 
                            help='Augment the training dataset')
    parser.add_argument('--gradient_clip_val', default=1.0, type=float, 
                            help='Clip gradients at specified value for stable training')

    # model parameters 
    parser.add_argument('--patch_size_one', default=1, type=int,
                         help='patch size for the channel space')
    parser.add_argument('--patch_size_two', default=100, type=int,
                        help='patch size for the time')
    parser.add_argument('--embed_dim', default=384, type=int, 
                        help='embedding dimension of the encoder')
    parser.add_argument('--depth', default=3, type=int, 
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
    parser.add_argument('--pixel_norm', default=False, type=bool, 
                        help='normalize the output pixels before computing the loss')
    
    # dataset parameters 
    parser.add_argument('--dataset_path', default='/data0/practical-sose23/brain-age/data', type=str, 
                        help='path that contains all the datasets')
    parser.add_argument('--dataset_version', default='v3.0', type=str, 
                        help='version of the preprocessed data')
    parser.add_argument('--train_dataset', default=['bap'], type=str, nargs='+', 
                        help='dataset for training. One of bap, hbn, lemon')
    parser.add_argument('--val_dataset', default=['bap'], type=str, nargs='+', 
                        help='dataset for validating. One of bap, hbn, lemon')
    parser.add_argument('--regressor_train_dataset', default=['bap'], type=str, nargs='+', 
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
    parser.add_argument('--crop_len', default=1000, type=int,
                       help='# of time samples of the random crop applied to the model input')
    parser.add_argument('--clamp_val', default=20, type=float, 
                        help='the input to the model will be limited between (-clamp_val, clamp_val)')
    parser.add_argument('--p_amplitude_flip', default=0.0, type=float,
                       help='probability of flipping the amplitude of the signal')
    parser.add_argument('--p_channels_dropout', default=0.0, type=float, 
                        help='probability of dropping a number of channels')
    parser.add_argument('--max_channels_to_dropout', default=0, type=int, 
                        help='maximum number of channels to dropout in channels_dropout')
    parser.add_argument('--p_gaussian_noise', default=0.0, type=float, 
                        help='probability for adding gaussian noise to the model input')


    # callbacks
    parser.add_argument('--checkpoint_callback', default=False, type=bool, 
                        help='to save best and last checkpoints')

    
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

    # Check for invalid arguments
    if "random_initialization" in args.mode:
        assert len(args.mode) == 1, "only one mode can be passed when training from random weights"
    assert len(args.mode) == len(args.epochs) == len(args.lr), "provide one learning rate and # epochs for each mode"   

    # To properly utilize 'NVIDIA A40' For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision('medium')
    # Use the selected device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Setting seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Download the model weights from wandb
    wandb.login()
    run = wandb.init()
    artifact_wandb_path = 'brain-age/brain-age/model-' + args.artifact_id
    artifact = run.use_artifact(artifact_wandb_path, type='model')
    artifact_path = pathlib.Path(artifact.download())
    checkpoint_path = list(artifact_path.rglob("*.ckpt"))[0]
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    run.finish()
    
    # Initialize and compose the transforms 
    if args.standardization == "channelwise":
        norm = channelwise_norm
    elif args.standardization == "channelwide":
        norm = channelwide_norm

    randomcrop = partial(_randomcrop, seq_len=args.crop_len)
    clamp = partial(_clamp, dev_val=args.clamp_val)
    transforms_train = [randomcrop, norm, clamp]
    transforms_val = [randomcrop, norm, clamp]

    if args.augment_data: # use data augmentation
        rand_amplitude_flip = partial(amplitude_flip, prob=args.p_amplitude_flip)
        rand_channels_dropout = partial(channels_dropout, max_channels=args.max_channels_to_dropout, prob=args.p_channels_dropout)
        rand_gaussian_noise = partial(gaussian_noise, prob=args.p_gaussian_noise)
        transforms_train += [rand_amplitude_flip, rand_channels_dropout, rand_gaussian_noise]

    composed_transforms_train = partial(_compose, transforms=transforms_train)
    composed_transforms_val = partial(_compose, transforms=transforms_val)

    # Initialize the training and validation dataloader
    train_dataset = EEGDataset(datasets_path=args.dataset_path, 
                                dataset_names=args.train_dataset, 
                                splits=['train'], 
                                d_version=args.dataset_version, 
                                transforms=composed_transforms_train)
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                pin_memory=True)

    val_dataset = EEGDataset(datasets_path=args.dataset_path, 
                                dataset_names=args.val_dataset, 
                                splits=['val'], 
                                d_version=args.dataset_version, 
                                transforms=composed_transforms_val, 
                                oversample=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                pin_memory=True, 
                                shuffle=True)

    # Configure callbacks
    callbacks = []
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

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
    
    print(f"\n===============================\n")
    print(args.mode, args.lr, args.epochs)
    print(f"\n===============================\n")
    
    # Perform fine tuning workflow
    for mode, lr, epochs in zip(args.mode, args.lr, args.epochs):
        
        wandb.login()
        logger = pl.loggers.WandbLogger(project="brain-age", name=f"{args.experiment_name}_{mode}_{args.artifact_id}", 
                                        save_dir="wandb/", log_model=False)
        
        if "finetune" in mode:
            head = copy.deepcopy(model.head)
        if mode == "random_initialization":
            state_dict = None
            mode = "finetune_encoder"

        model = VisionTransformer(state_dict=state_dict, \
                                            img_size=(args.num_channels, args.input_time * args.fs), \
                                            patch_size=(args.patch_size_one, args.patch_size_two), \
                                            in_chans=1, 
                                            embed_dim=args.embed_dim, 
                                            depth=args.depth, 
                                            num_heads=args.num_heads, 
                                            mlp_ratio=args.mlp_ratio, 
                                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                            mode=mode,
                                            max_epochs=epochs,
                                            lr=lr)
        if "finetune" in mode and not "random_initialization" in args.mode:
            model.head = head

        trainer = pl.Trainer(overfit_batches=args.overfit_batches,  # for fast debugging
                            deterministic=True, # for reproducibility
                            devices=[0], 
                            callbacks=callbacks,
                            check_val_every_n_epoch=1,
                            max_epochs=epochs, 
                            accelerator="gpu", 
                            logger=logger,
                            precision="bf16-mixed", # for speed
                            gradient_clip_val=args.gradient_clip_val)
        print(f"\n===============================\n")
        print(f"Training with mode: {mode}")
        print(f"\n===============================\n")

        trainer.fit(
            model=model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader
            )
        wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)