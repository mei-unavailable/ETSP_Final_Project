import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,  padding='same'):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.
    Args:
        inputs (tensor : B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        kernelSD (float): Standard deviation of the Gaussian smoothing kernel.
        padding (str): Padding mode, either 'same' or 'valid'.
        device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        smoothed (tensor : B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Get Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # Convert to tensor
    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution
    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
    return smoothed.permute(0, 2, 1)  # [B, T, C]


def spec_augment(
    inputs,
    time_mask_param=0,
    time_mask_count=0,
    freq_mask_param=0,
    freq_mask_count=0,
    mask_value=0.0,
):
    """Apply simple SpecAugment-style masking on neural features.

    Args:
        inputs: Tensor of shape (B, T, C).
        time_mask_param: Max width of each time mask.
        time_mask_count: Number of time masks to apply.
        freq_mask_param: Max width of each feature/channel mask.
        freq_mask_count: Number of feature masks to apply.
        mask_value: Value to fill masked regions with.
    """

    if time_mask_param <= 0 and freq_mask_param <= 0:
        return inputs

    B, T, C = inputs.shape
    device = inputs.device
    x = inputs

    if time_mask_param > 0 and time_mask_count > 0:
        for _ in range(time_mask_count):
            mask_len = torch.randint(0, time_mask_param + 1, (1,), device=device).item()
            if mask_len == 0:
                continue
            start = torch.randint(0, max(T - mask_len, 1), (1,), device=device).item()
            end = min(start + mask_len, T)
            x[:, start:end, :] = mask_value

    if freq_mask_param > 0 and freq_mask_count > 0:
        for _ in range(freq_mask_count):
            mask_len = torch.randint(0, freq_mask_param + 1, (1,), device=device).item()
            if mask_len == 0:
                continue
            start = torch.randint(0, max(C - mask_len, 1), (1,), device=device).item()
            end = min(start + mask_len, C)
            x[:, :, start:end] = mask_value

    return x