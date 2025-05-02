"""
Created on Tue Apr 22 12:18:44 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import arange, floor, tensor, mean, sqrt
from torch import abs as abs_torch
# =============================================================================
# =============================================================================


def get_freq_grid(nch, spacing):
    """
    Compute a symmetric frequency grid for WDM channels.

    Args
    ----
        nch (int): Number of WDM channels.
        spacing (float): Channel spacing (Hz, GHz, etc.).

    Returns
    -------
        torch.Tensor: 1D tensor of length 'nch' containing the center
                      frequencies, symmetric around zero.
    """
    nch = tensor(nch)

    if (nch % 2) == 0:

        # central frequencies of the WDM channels
        freqGrid = arange(-floor(nch / 2), floor(nch / 2), 1) * spacing

        freqGrid += spacing / 2

    else:

        # central frequencies of the WDM channels
        freqGrid = arange(-floor(nch / 2), floor(nch / 2) + 1, 1) * spacing

    return freqGrid


def p_corr(x, y):

    x_0 = x - mean(x, dim=-1, keepdim=True)
    y_0 = y - mean(y, dim=-1, keepdim=True)

    # Denominator of Pearson correlation
    p_coeff = mean(x_0 * y_0.conj(), dim=-1) / \
        sqrt(mean(abs_torch(x_0) ** 2, dim=-1) *
             mean(abs_torch(y_0) ** 2, dim=-1))

    return abs_torch(p_coeff)
