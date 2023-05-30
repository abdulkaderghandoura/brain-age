import torch
import numpy as np

def _compose(x, transforms):
    for transform in transforms:
        x = transform(x)
    return x

def _randomcrop(x, seq_len):
    idx = torch.randint(low=0, high=x.shape[-1]-seq_len, size=(1,))
    return x[:, idx:idx+seq_len]

def totensor(x):
    return torch.tensor(x).float()

def channelwide_norm(x, eps=1e-8):
    return (x - x.mean()) / x.std()

def channelwise_norm(x, eps=1e-8):
    return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)

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

