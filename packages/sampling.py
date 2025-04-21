"""
Created on Mon Apr 21 19:15:56 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import zeros, zeros_like, cfloat, float32, arange, pi, isclose, \
    tensor, sqrt, sin, cos, mean
from torch import abs as abs_torch
from torch.nn.functional import conv1d
# =============================================================================
# =============================================================================


def up_sampling(qam_tensor, k, device):
    """
    Perform zero-insertion upsampling along the last dimension of a tensor.

    Inserts (k - 1) zeros between each element along the last dimension,

    Args
    ----
        qam_tensor (torch.Tensor): Input tensor of shape (..., n).
        k (int): Upsampling factor. Must be >= 1.

    Returns
    -------
        torch.Tensor: Upsampled tensor (..., n * k), with zeros inserted.

    Example
    -------
        >>> x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        >>> up = up_sampling(x, k=3)
        >>> print(up)
        tensor([[1., 0., 0., 2., 0., 0.],
                [3., 0., 0., 4., 0., 0.]])
    """
    *leading_dims, n = qam_tensor.shape
    out_shape = (*leading_dims, n * k)

    # Create output tensor of zeros
    out = zeros(out_shape, dtype=cfloat, device=device)

    # Create an index to place original data at every k-th position
    idx = arange(n, device=device) * k
    out[..., idx] = qam_tensor

    return out


def rrc_filter(alpha, symbs, k, device):
    """
    Create a Root Raised Cosine (RRC) filter impulse response using PyTorch.

    Args
    ----
        alpha (float): Roll-off factor (0 < alpha <= 1).
        symbs (int): Filter length in symbols (e.g., 8).
        k (int): Upsampling factor (samples per symbol).
        device (torch.device): The device for the output tensor.

    Returns
    -------
        torch.Tensor: 1D tensor of shape (symbs * k_up + 1,) containing RRC
                      filter taps.

    Example
    -------
        >>> rrc = rrc_filter(alpha=0.25, symbs=8, k_up=4)
        >>> print(rrc.shape)  # torch.Size([33])
    """
    # Time index k (symmetrical around 0)
    N = symbs * k
    t = arange(-N / 2, N / 2 + 1, dtype=float32, device=device) / k

    # small epsilon to avoid division by zero
    eps = 1e-8

    # RRC impulse response initialization with zeros
    g = zeros_like(t)

    # Indices for singularities
    i1 = isclose(t, tensor(0., dtype=float32, device=device), atol=eps)
    i2 = isclose(abs_torch(4 * alpha * t), tensor(1., dtype=float32,
                                                  device=device), atol=eps)
    # everything else
    i3 = ~(i1 | i2)

    # k == 0 (center)
    g[i1] = 1 - alpha + 4 * alpha / pi

    # k == ±1/(4*alpha)
    if alpha > 0:
        g[i2] = (alpha / sqrt(tensor(2))) * (
            (1 + 2 / pi) * sin(tensor(pi / (4 * alpha))) +
            (1 - 2 / pi) * cos(tensor(pi / (4 * alpha)))
        )

    # General case
    k = t[i3]

    numerator = sin(pi * k * (1 - alpha)) + 4 * alpha * k * \
        cos(pi * k * (1 + alpha))

    denominator = pi * k * (1 - (4 * alpha * k) ** 2)

    g[i3] = numerator / (denominator + eps)

    # Normalize filter to unit peak
    g = g / g.max()

    return g


def shaping_filter(data, filter_coeffs, device):
    """
    Apply a 1D shaping filter along the last dimension of a tensor.

    The filter is applied using 1D convolution and the output is normalized to
    unit average power.

    Args
    ----
        data (torch.Tensor): Input tensor of shape (n_ch, n_pol, n_samples).
        filter_coeffs (torch.Tensor): 1D tensor of filter coefficients,
                                      shape (filter_len,).

    Returns
    -------
        torch.Tensor: Filtered and normalized tensor of shape
                      (n_ch, n_pol, n_samples + filter_len - 1).
    """
    n_ch, n_pol, n_samples = data.shape
    filter_len = filter_coeffs.numel()

    # Prepare for conv1d: reshape to (n_ch * n_pol, 1, n_samples)
    data_reshaped = data.view(n_ch * n_pol, 1, n_samples)

    # Prepare filter: shape (1, 1, filter_len), flipped for conv1d
    filt = filter_coeffs.to(dtype=cfloat, device=device).flip(0).view(1, 1, -1)

    # Apply 'full' convolution by padding the left side
    padding = filter_len - 1

    # Perform convolution
    out = conv1d(data_reshaped, filt, padding=padding)

    # Reshape back to (n_ch, n_pol, new_n_samples)
    new_n_samples = out.shape[-1]
    out = out.view(n_ch, n_pol, new_n_samples)

    # Normalize to unit average power
    power = mean(abs_torch(out) ** 2, dim=-1, keepdim=True)
    out = out / sqrt(power + 1e-8)

    return out


def matched_filter(data, filter_coeffs, n_symbs, k, device):
    """
    Apply a 1D matched filter along the last dimension of a tensor.

    The filter is applied using 1D convolution and the output is normalized to
    unit average power.

    Args
    ----
        data (torch.Tensor): Input tensor of shape (n_ch, n_pol, n_samples).
        filter_coeffs (torch.Tensor): 1D tensor of filter coefficients,
                                      shape (filter_len,).

    Returns
    -------
        torch.Tensor: Filtered and normalized tensor of shape
                      (n_ch, n_pol, n_samples + filter_len - 1).
    """
    n_ch, n_pol, n_samples = data.shape
    filter_len = filter_coeffs.numel()

    # Prepare for conv1d: reshape to (n_ch * n_pol, 1, n_samples)
    data_reshaped = data.view(n_ch * n_pol, 1, n_samples)

    # Prepare filter: shape (1, 1, filter_len), flipped for conv1d
    filt = filter_coeffs.to(dtype=cfloat, device=device).flip(0).view(1, 1, -1)

    # Apply 'full' convolution by padding the left side
    padding = filter_len - 1

    # Perform convolution
    out = conv1d(data_reshaped, filt, padding=padding)

    # Reshape back to (n_ch, n_pol, new_n_samples)
    new_n_samples = out.shape[-1]
    out = out.view(n_ch, n_pol, new_n_samples)

    # Exclude transient samples
    out = out[..., int(n_symbs * k):-int(n_symbs * k)]

    # Normalize to unit average power
    power = mean(abs_torch(out) ** 2, dim=-1, keepdim=True)
    out = out / sqrt(power + 1e-8)

    return out
