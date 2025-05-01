"""
Created on Thu May  1 15:58:16 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import float32, arange, argmax, exp, pi
from torch.fft import fft, fftshift
from torch import abs as abs_torch
# =============================================================================
# =============================================================================


def freq_rec_4th_power(signal, sr, eq_convergence, device):
    """
    4th-power frequency recovery method.

    This method raises the received signal to the 4th power, computes its FFT,
    and identifies the dominant frequency component corresponding to 4× the
    true carrier offset. It then corrects the original signal by applying a
    complex exponential with the estimated offset.

    Args
    ----
        signal (torch.Tensor): Complex input signal (n_ch, n_pol, n_samples)
        sr (float): Symbol rate [Hz]
        eq_convergence (int): Number of initial symbols to skip due to
        equalizer convergence
        device (torch.device): PyTorch device

    Returns
    -------
        torch.Tensor: Frequency-corrected signal (same shape as input)

    Notes
    -----
        - Assumes 1 sample per symbol (symbol-spaced signal)
        - Best suited for 4-QAM, 16-QAM, etc. (square QAM constellations)
        - 'signal ** 4' removes modulation, revealing the frequency offset
    """
    # Get the number of channels, polarizations, and samples
    n_ch, n_pol, n_s = signal.shape

    # Discrete time indexes
    n = arange(0, n_s, dtype=float32, device=device)

    # Adjust n_s to not take into account the equalizer convergence
    n_s = n_s - eq_convergence

    # Frequency vector considering 1 sample per symbol and symbol period:
    f = arange(-1 / 2, 1 / 2, 1 / n_s, dtype=float32, device=device) * sr

    # Obtaining the absolute value of the spectrum of the signal^4:
    signal_spectrum = fftshift(abs_torch(fft(
                                           signal[..., eq_convergence:] ** 4)))

    # Obtaining the frequency offset:
    delta_f = (1 / 4) * f[argmax(signal_spectrum, dim=-1, keepdim=True)]

    # Apply the frequency correction
    out = signal * exp(-2j * pi * delta_f * n / sr)

    return out
