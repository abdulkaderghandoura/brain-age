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
import copy
from transforms import channelwide_norm, channelwise_norm, _clamp, _randomcrop, _compose, amplitude_flip, channels_dropout, time_masking, gaussian_noise
from dataset import EEGDataset
from mae import MaskedAutoencoderViT

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import numpy as np
import pathlib
# torch.cuda.empty_cache() 
from mae_age_regressor import MAE_AGE
from mae_finetuner import VisionTransformer

def get_args_parser():
    parser = argparse.ArgumentParser('MAE training', add_help=False)
    
    parser.add_argument('--experiment_name', default='downstream_task')
    
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=[80, 150], type=int, nargs='+', 
                        help='Maximum number of epochs for linear probing and finetuning, respectively')

    parser.add_argument('--train_dataset', default=['bap'], type=list, nargs='+', 
                        help='dataset for training eg. bap, hbn, lemon')
    
    parser.add_argument('--val_dataset', default=['bap'], type=list, nargs='+', 
                        help='dataset for training eg. bap, hbn, lemon')
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
    
    parser.add_argument('--p_time_masking', default=0.0, type=float,
                       help='# of time samples of the random crop applied to the model input')
    
    parser.add_argument('--max_mask_size', default=0, type=int,
                       help='# of time samples masked across all channels')
    
    parser.add_argument('--p_gaussian_noise', default=0.0, type=float, 
                        help='probability for adding gaussian noise to the model input')
    # model parameters 
    parser.add_argument('--input_time', default=10, type=int,
                        help='number of seconds in the input')

    parser.add_argument('--patch_size', default=100, type=int, # number of patches = 30s * 135 / 90 (in the case we are using patch_size[0] = 65)
                        help='patch input size')
    
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # to avoid a bottleneck 256 which is the number of cpus on the machine
    parser.add_argument('--num_workers', default=10, type=int, 
                        help='number of workers for the dataloaders')
    
    parser.add_argument('--embed_dim', default=384, type=int, 
                        help='embedding dimension of the encoder')
    parser.add_argument('--depth', default=3, type=int, 
                        help='the number of blocks of the encoder')
    parser.add_argument('--num_heads', default=6, type=int, 
                        help='the number of attention heads of the encoder')
    parser.add_argument('--decoder_embed_dim', default=256, type=int, 
                        help='the embedding dimension of the decoder')
    parser.add_argument('--decoder_depth', default=2, type=int, 
                        help='the number of blocks of the decoder')
    parser.add_argument('--decoder_num_heads', default=8, type=int, 
                    help='number of attention heads of the decoder')
    parser.add_argument('--mlp_ratio', default=4, type=int, 
                        help='ratio of mlp hidden dim to embedding dim')
    
    
    parser.add_argument('--mae_age', default=True, type=bool, 
                        help='run mae with age regression head or not')
    parser.add_argument('--oversample', default=False, type=bool, 
                        help='to oversample the minority dataset when training on target and external dataset ')
    parser.add_argument('--overfit_batches', default=1.0, type=float, 
                        help='to debug model on a fraction of batches')
    
    parser.add_argument('--lr', default=[1e-2, 1e-4], type=float, nargs='+', 
                        help='learning rates for linear probing and finetuning, respectively')

    parser.add_argument('--artifact_id', default='h4vidwgf:v210', type=str, 
                        help='name and version of the model artifact to be finetuned') # for lemon: 0q3jg1cd:v0
    
    parser.add_argument('--reinitialize_weights', default=False, type=bool, 
                        help='reinitialize the weights randomly as a control')
    parser.add_argument('--mode', default=["linear_probe", "finetune_encoder"] , type=str, nargs='+',
                        help='select mode to fine tune: linear_probe, finetune_encoder, finetune_final_layer, random_initialization')
    parser.add_argument('--augment_data', default=False, type=bool, 
                            help='augment the training dataset')

    return parser
    


def main(args):

    # using a CUDA device ('NVIDIA A80') that has Tensor Cores. 
    # To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
    # which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    
    torch.set_float32_matmul_precision('medium')

    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    #size of the input = # of seconds * sampling frequency 
    
    wandb.login()
    run = wandb.init()
    artifact_wandb_path = 'brain-age/brain-age/model-' + args.artifact_id
    artifact = run.use_artifact(artifact_wandb_path, type='model')
    artifact_path = pathlib.Path(artifact.download())
    ckpt_path = list(artifact_path.rglob("*.ckpt"))[0]
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    run.finish()
    
    if args.standardization == "channelwise":
        norm = channelwise_norm
    elif args.standardization == "channelwide":
        norm = channelwide_norm

    randomcrop = partial(_randomcrop, seq_len=args.crop_len)
    clamp = partial(_clamp, dev_val=args.clamp_val)
    transforms_train = [randomcrop, norm, clamp]
    transforms_val = [randomcrop, norm, clamp]

    if args.augment_data:
        rand_amplitude_flip = partial(amplitude_flip, prob=args.p_amplitude_flip)
        rand_channels_dropout = partial(channels_dropout, max_channels=args.max_channels_to_dropout, prob=args.p_channels_dropout)
        rand_gaussian_noise = partial(gaussian_noise, prob=args.p_gaussian_noise)
        transforms_train += [rand_amplitude_flip, rand_channels_dropout, rand_gaussian_noise]

    composed_transforms_train = partial(_compose, transforms=transforms_train)
    composed_transforms_val = partial(_compose, transforms=transforms_val)

    train_dataset = EEGDataset(datasets_path="/data0/practical-sose23/brain-age/data/", dataset_names=['bap'], splits=['train'], d_version="v3.0", transforms=composed_transforms_train, oversample=True)
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                pin_memory=True, 
                                shuffle=True)

    val_dataset = EEGDataset(datasets_path="/data0/practical-sose23/brain-age/data/", dataset_names=['bap'], splits=['val'], d_version="v3.0", transforms=composed_transforms_val, oversample=False)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                pin_memory=True, 
                                shuffle=True)
    print(f"\n===============================\n")
    print(args.mode, args.lr, args.epochs)
    print(f"\n===============================\n")
    
    if "random_initialization" in args.mode:
        assert len(args.mode) == 1, "only one mode can be passed when training from random weights"

    for mode, lr, epochs in zip(args.mode, args.lr, args.epochs):
        
        wandb.login()
        logger = pl.loggers.WandbLogger(project="brain-age", name=f"{args.experiment_name}_{mode}_{args.artifact_id}", 
                                        save_dir="wandb/", log_model=False)
        
        print(f"\n===============================\n")
        print("mode", mode)
        print("epochs", epochs)
        print("lr", lr)
        print(f"\n===============================\n")

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        if "finetune" in mode:
            head = copy.deepcopy(model.head)
        if mode == "random_initialization":
            state_dict = None
            mode = "finetune_encoder"

        model = VisionTransformer(state_dict=state_dict, \
                                            img_size=(63, args.input_time * 100), \
                                            patch_size=(1, 100), \
                                            in_chans=1, 
                                            embed_dim=args.embed_dim, 
                                            depth=args.depth, 
                                            num_heads=args.num_heads, 
                                            mlp_ratio=args.mlp_ratio, 
                                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                            mode=mode,
                                            max_epochs=epochs,
                                            lr=lr)
        if "finetune" in mode:
            model.head = head

        trainer = pl.Trainer(
                            deterministic=True, # to ensure reproducibility 
                            devices=[0], 
                            callbacks=[lr_monitor,                         
                            ], 
                            check_val_every_n_epoch=1,
                            max_epochs=epochs, 
                            accelerator="gpu", 
                            logger=logger,
                            precision="bf16-mixed", 
                            gradient_clip_val=1
                            )
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