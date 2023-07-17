import numpy as np
import torch

def _compose(x, transforms):
    for transform in transforms:
        x = transform(x)
    return x

def _randomcrop(x, seq_len):
    idx = torch.randint(low=0, high=x.shape[-1]-seq_len, size=(1,))
    return x[:, :, idx:idx+seq_len]

def totensor(x):
    return torch.tensor(x).float()

def channelwide_norm(x, eps=1e-8):
    return (x - x.mean()) / (eps + x.std())

def channelwise_norm(x, eps=1e-8):
    return (x - x.mean(-1, keepdims=True)) / (eps + x.std(-1, keepdims=True))

def _clamp(x, dev_val):
    """input: normalized"""
    return torch.clamp(x, -dev_val, dev_val)

def toimshape(x):
    return x.unsqueeze(0)

def _labelcenter(y, mean_age):
    return y - mean_age

def _labelnorm(y, mean_age, std_age, eps=1e-6):
    return (y - mean_age) / torch.sqrt(std_age**2 + eps)

def _labelbin(y, y_lower):
    return int(y > y_lower)

def _labelmultibin(y, y_lower):
    return torch.where(y >= y_lower)[0].max()

def gaussian_noise(x, prob, mean=0, std=1e-5):
    """Applies Gaussian noise to the input tensor with a given probability.

    Args:
        x (torch.Tensor): The input tensor.
        prob (float): The probability of applying Gaussian noise.
        mean (float, optional): The mean of the Gaussian noise. Defaults to 0.
        std (float, optional): The standard deviation of the Gaussian noise. Defaults to 1e-5.

    Returns:
        torch.Tensor: The input tensor with Gaussian noise applied (if selected) or the original tensor.
    """
    x_augmented = x.clone()
    if torch.rand(1) < prob:
        # Generate Gaussian noise with the same shape as x
        noise = torch.randn_like(x) * std + mean
        x_augmented = x_augmented + noise
    
    return x_augmented

def channels_dropout(x, prob, max_channels):
    """Applies channel dropout to the input tensor with a given probability.

    Args:
        x (torch.Tensor): The input tensor.
        prob (float): The probability of applying channel dropout.
        max_channels (int): The maximum number of channels to be dropped out.

    Returns:
        torch.Tensor: The input tensor with channel dropout applied (if selected) or the original tensor.
    """
    x_augmented = x.clone()
    if torch.rand(1) < prob:
        # Generate a random number of channels to be dropped out
        num_channels = torch.randint(1, max_channels + 1, (1,))
        # Generate unique random indices of channels to be dropped out
        dropout_channels = torch.randperm(x.size(0))[:num_channels]
        # Set the corresponding channel values to 0
        x_augmented[dropout_channels, :] = 0
    
    return x_augmented


def time_masking(x, prob, max_mask_size, mode='same_segment'):
    """Applies time masking to the input tensor with a given probability.

    Args:
        x (torch.Tensor): The input tensor.
        prob (float): The probability of applying time masking.
        max_mask_size (int): The maximum size of the mask.
        mode (str, optional): The mode of time masking. Defaults to 'same_segment'.
            - 'same_segment': Masks the same segment from all channels.
            - 'random_segment': Masks a random segment from each channel.

    Returns:
        torch.Tensor: The input tensor with time masking applied (if selected) or the original tensor.
    """
    x_augmented = x.clone()
    if torch.rand(1) < prob:
        # Mask the same segment from all channels
        if mode == 'same_segment':
            # Generate a random mask size within the given range
            mask_size = torch.randint(1, max_mask_size + 1, (1,))
            # Generate a random start index for masking
            start = torch.randint(0, x.size(1) - int(mask_size), (1,))
            end = start + mask_size
            assert 0 <= start < end <= x.size(1)
            x_augmented[:, start:end] = 0
        
        # Mask a random segment from each channel
        elif mode == 'random_segment':
            for c in range(x.size(0)):
                mask_size = torch.randint(1, max_mask_size + 1, (1,))
                start = torch.randint(0, x.size(1) - int(mask_size), (1,))
                end = start + mask_size
                assert 0 <= start < end <= x.size(1)
                x_augmented[c, start:end] = 0

    return x_augmented

def amplitude_flip(x, prob):
    """Applies amplitude flipping to the input tensor with a given probability.

    Args:
        x (torch.Tensor): The input tensor.
        prob (float): The probability of applying amplitude flipping.

    Returns:
        torch.Tensor: The input tensor with amplitude flipping applied (if selected) or the original tensor.
    """
    if torch.rand(1) < prob:
        x = -x
    
    return x
