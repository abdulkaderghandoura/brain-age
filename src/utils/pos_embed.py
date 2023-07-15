# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# References: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------
# Changes made to the original implementation: 
#       - adjusted the implementation to work on non square input 
import numpy as np

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generates 2D sine-cosine positional embeddings for a given grid size.

    Args:
        embed_dim (int): Dimensionality of the positional embeddings.
        grid_size (tuple): Tuple of two integers representing the grid height and width.
        cls_token (bool, optional): Whether to include a classification token in the positional embeddings. 
                                    Defaults to False.

    Returns:
        pos_embed (numpy.ndarray): 2D positional embeddings with shape [grid_size*grid_size, embed_dim] or 
                                   [1+grid_size*grid_size, embed_dim] if cls_token is True.
    """

    grid_size_h = grid_size[0]
    grid_size_w = grid_size[1]
    grid_h = np.arange(grid_size_h, dtype=np.float32) # grid size = self.patch_embed.num_patches**.5
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first

    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_w, grid_size_h]) # shape before (2, 14, 14) shape after (2, 1, 14, 14)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid) # shape of the pos embed (196, 1024) for the encoder 
    # shape of the pos embed for the decoder is (196, 512)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):

    """
    Generates 2D sine-cosine positional embeddings from a grid.

    Args:
        embed_dim (int): Dimensionality of the positional embeddings. Must be divisible by 2.
        grid (numpy.ndarray): 2D grid representing the positions.

    Returns:
        emb (numpy.ndarray): 2D positional embeddings with shape (H*W, D), where H is the grid height, 
                            W is the grid width, and D is the embed_dim.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    h_embed_dim = embed_dim // 2 # embed dim = 1024 for the encoder 
    w_embed_dim = embed_dim // 2 
    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generates 1D sine-cosine positional embeddings from a list of positions.

    Args:
        embed_dim (int): Output dimension for each position. Must be divisible by 2.
        pos (numpy.ndarray): List of positions to be encoded. Shape: (M,)

    Returns:
        emb (numpy.ndarray): 1D positional embeddings with shape (M, D), where M is the number of positions
                             and D is the embed_dim.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32) # 0 --> 256 
    omega /= embed_dim / 2. #( 0 --> 256  / 256)
    omega = 1. / 10000**omega  # (D/2,) 
    # omega = 1/10000**i --> i = range_normalized(0,1) for embedding = 1024 total and 512 for each side of the grid 

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
