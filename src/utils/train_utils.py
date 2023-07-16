import wandb 
import torch 
import matplotlib.pyplot as plt
def visualize(num_samples, patch_size, mask, target, pred, split="train"): 
        """
        Visualizes the target, reconstructed, masked, and mask signals.

        Args:
            num_samples(int): number of time samples in one channel 
            mask (torch.Tensor): Binary mask tensor of shape (N, L), where 0 represents elements to keep
                and 1 represents elements to remove.
            target (torch.Tensor): Target tensor of shape (N, L, p1*p2).
            pred (torch.Tensor): Reconstructed tensor of shape (N, L, p1*p2).
            split (str, optional): Name of the split (e.g., "train", "val"). Defaults to "train".

        """
        # Apply the mask on the target 
        inverted_mask = 1 - mask
        expanded_mask = inverted_mask[0].unsqueeze(-1)
        masked_target = target[0] * expanded_mask
        
        # Extract one channel from target and pred 
        ch_idx = 0
        target_channel = []
        masked_channel = []
        pred_channel = []
        mask_channel = []
        for patch_idx in range(target.shape[1]):
            pred_patch = pred[0, patch_idx, :].view(patch_size)
            target_patch = target[0, patch_idx, :].view(patch_size)
            # Extract channel of interest from target patch and predicted patch 
            target_channel.append(target_patch[ch_idx, :])
            pred_channel.append(pred_patch[ch_idx, :])
            # Apply mask to target channel
            masked_channel.append(target_patch[ch_idx, :] * (1-mask[0, patch_idx]))
        # Concatenate target channel patches
        target_channel = torch.cat(target_channel)
        # Concatenate predicted channel patches
        pred_channel = torch.cat(pred_channel)
        # Concatenate masked channel patches
        masked_channel = torch.cat(masked_channel)
        mask_channel = masked_channel == 0

        # Plotting the signals
        fig, ax = plt.subplots(4, figsize=(15, 5))
        ax[0].plot(target_channel.cpu().float()[:num_samples]) # plot target channel signal
        ax[0].set_title("target")
        ax[1].plot(pred_channel.cpu().float().detach().numpy()[:num_samples]) # plot reconstructed channel signal
        ax[1].set_title("reconstruction")
        ax[2].plot(masked_channel.cpu().float()[:num_samples])  # plot masked channel signal
        ax[2].set_title("masked")
        ax[3].plot(mask_channel.cpu().float()[:num_samples]) # plot binary mask channel signal
        ax[3].set_title("mask")
        fig.tight_layout()
        wandb.log({"signals_{}".format(split): fig})



