"""
Created on Sun May  4 11:23:28 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import where, mean, float32
from torch import abs as abs_torch
# =============================================================================
# =============================================================================


def ber_comp(bit_tx, bit_rx):
    """
    Compute the BER between transmitted and received bit tensors.

    Parameters
    ----------
    bit_tx : torch.Tensor
        Transmitted bits of shape (..., n_bits), where the last dimension is
        the bit sequence.

    bit_rx : torch.Tensor
        Received bits of the same shape as bit_tx.

    Returns
    -------
    ber : torch.Tensor
        Bit Error Rate computed over the last dimension. Output shape is the
        same as input excluding the last dimension.
    """
    ber = mean(abs_torch(bit_tx - bit_rx), dim=-1, dtype=float32)

    return ber


def ser_comp(symb_tx, symb_rx):
    """
    Compute the SER between transmitted and received symbol tensors.

    Parameters
    ----------
    symb_tx : torch.Tensor
        Transmitted symbols of shape (..., n_symbols), where the last dimension
        is the symbol sequence.

    symb_rx : torch.Tensor
        Received symbols of the same shape as symb_tx.

    Returns
    -------
    ser : torch.Tensor
        Symbol Error Rate computed over the last dimension. Output shape is the
        same as input excluding the last dimension.
    """
    ser = mean(where(symb_tx != symb_rx, 1.0, 0.0), dim=-1)

    return ser
