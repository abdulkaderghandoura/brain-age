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
import pathlib
# torch.cuda.empty_cache() 
from mae_age_regressor import MAE_AGE
from mae_finetuner import VisionTransformer

def get_args_parser():
    parser = argparse.ArgumentParser('MAE training', add_help=False)
    parser.add_argument('--experiment_name', default='finetuning')
    
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=400, type=int)

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
    
    # model parameters 
    parser.add_argument('--input_time', default=10, type=int,
                        help='number of seconds in the input')

    parser.add_argument('--patch_size', default=90, type=int, # number of patches = 30s * 135 / 90 (in the case we are using patch_size[0] = 65)
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
    
    parser.add_argument('--lr_mae', default=2.5e-4, type=float, 
                        help='learning rate to train the masked autoencoder with')    
    parser.add_argument('--lr_regressor', default=2.5e-4, type=float, 
                        help='learning rate to train the regression head with')
    
    parser.add_argument('--artifact_id', default='68ww7y5i:v19', type=str, 
                        help='name and version of the model artifact to be finetuned') # for lemon: 0q3jg1cd:v0
    
    parser.add_argument('--reinitialize_weights', default=False, type=bool, 
                        help='reinitialize the weights randomly as a control')
    parser.add_argument('--finetune_mode', default="linear_probe", type=str, 
                        help='select mode to fine tune different parts of the architecture: linear_probe, finetune_encoder')
    
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
    def get_encoder_checksum(encoder):
        checksum = 0
        for params in encoder.parameters():
            checksum += params.sum()
        return checksum



#     if args.mae_age:
#         model = MAE_AGE(img_size=(61, args.input_time * 100), \
#                                     patch_size=(1, 100), \
#                                     in_chans=1, 
#                                     embed_dim=args.embed_dim, 
#                                     depth=args.depth, 
#                                     num_heads=args.num_heads, 
#                                     decoder_embed_dim=args.decoder_embed_dim, 
#                                     decoder_depth=args.decoder_depth, 
#                                     decoder_num_heads=args.decoder_num_heads,
#                                     mlp_ratio=args.mlp_ratio, 
#                                     norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
# #                                         lr_mae=args.lr_mae,
# #                                         lr_regressor=args.lr_regressor
#                                     # norm_pix_loss=True
#                                     )
#     else:
#         model = MaskedAutoencoderViT(img_size=(61, args.input_time * 100), \
#                                             patch_size=(1, 100), \
#                                             in_chans=1, 
#                                             embed_dim=args.embed_dim, 
#                                             depth=args.depth, 
#                                             num_heads=args.num_heads, 
#                                             decoder_embed_dim=args.decoder_embed_dim, 
#                                             decoder_depth=args.decoder_depth, 
#                                             decoder_num_heads=args.decoder_num_heads,
#                                             mlp_ratio=args.mlp_ratio, 
#                                             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#     #                                         lr_mae=args.lr_mae,
#     #                                         lr_regressor=args.lr_regressor
#                                             # norm_pix_loss=True
#                                             )
                                    
    if args.reinitialize_weights:
        wandb.login()
        pass
    else:
        wandb.login()
        run = wandb.init()
        artifact_wandb_path = 'brain-age/brain-age/model-' + args.artifact_id
        artifact = run.use_artifact(artifact_wandb_path, type='model')
        artifact_path = pathlib.Path(artifact.download())
        ckpt_path = list(artifact_path.rglob("*.ckpt"))[0]
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cuda:0'))
        checkpoint_model = checkpoint['state_dict']

        # from mae import interpolate_pos_embed
        # interpolate position embedding
        # interpolate_pos_embed(model, checkpoint_model)
        # msg = model.load_state_dict(checkpoint_model, strict=False)
        # print("------------->", msg)
        run.finish()

    # model = MAE_Finetuner(model, args.lr_mae, args.finetune_mode)
    
    
    model = VisionTransformer(state_dict=checkpoint_model, \
                                    img_size=(63, args.input_time * 100), \
                                    patch_size=(1, 100), \
                                    in_chans=1, 
                                    embed_dim=args.embed_dim, 
                                    depth=args.depth, 
                                    num_heads=args.num_heads, 
                                    # decoder_embed_dim=args.decoder_embed_dim, 
                                    # decoder_depth=args.decoder_depth, 
                                    # decoder_num_heads=args.decoder_num_heads,
                                    mlp_ratio=args.mlp_ratio, 
                                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#                                         lr_mae=args.lr_mae,
#                                         lr_regressor=args.lr_regressor
                                    # norm_pix_loss=True
                                    )
    print(f"========= \n checksum outside finetuner: {get_encoder_checksum(model.backbone.blocks)}")
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
    
    
    train_dataset = EEGDataset(['hbn', 'bap'], ['train'], transforms=composed_transforms, oversample=True)
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                pin_memory=True, 
                                shuffle=True)

    val_dataset = EEGDataset(['bap'], ['val'], transforms=composed_transforms, oversample=False)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                pin_memory=True, 
                                shuffle=True)
    
    wandb.login()
    logger = pl.loggers.WandbLogger(project="brain-age", name=args.experiment_name+"_"+args.artifact_id, 
                                    save_dir="wandb/", log_model=False)
    
    # early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-7, patience=3, verbose=False, mode="min")

    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',  # Metric to monitor for saving the best model
    #     filename='best_model',  # Filename pattern for saved models
    #     save_top_k=1,  # Number of best models to save (set to 1 for the best model only)
    #     mode='min',  # Mode of the monitored metric (minimize val_loss in this case)
    #     dirpath='../../models/checkpoints/{}'.format(args.experiment_name),
    #     save_last=True
    # )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')


    trainer = pl.Trainer(
                        # overfit_batches=args.overfit_batches,
                        deterministic=True, # to ensure reproducibility 
                        devices=[0], 
                        callbacks=[lr_monitor,                         
                        # checkpoint_callback, 
                        # early_stop_callback
                        ], 
                        # accumulate_grad_batches=32,
                        check_val_every_n_epoch=1,
                        max_epochs=args.epochs, 
                        accelerator="gpu", 
                        logger=logger,
                        precision="bf16-mixed", 
                        # fast_dev_run=True, 
                        )
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