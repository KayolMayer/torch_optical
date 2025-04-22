"""
Created on Tue Apr 22 11:30:53 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import manual_seed, float32, arange, sqrt, ones, pi, tensor, exp, \
    randn, cumsum
# =============================================================================
# =============================================================================


def laser_tx(n_ch, n_pol, n_s, power_dbm, sr, k, lw, f_grid, seed, device):
    """
    Generate a multi-channel optical laser signal with optional phase noise.

    Args
    ----
        n_ch (int): Number of channels.
        n_pol (int): Number of polarizations (1 or 2).
        n_s (int): Number of samples.
        power_dbm (float): Power per channel in dBm.
        sr (float): Symbol rate in Hz.
        k (int): Oversampling factor.
        lw (float): Laser linewidth (Hz). If > 0, phase noise is added.
        f_grid (torch.Tensor): 1D tensor of center frequencies per
                               channel (Hz), shape (n_ch,).
        seed (int): Random seed for reproducibility.
        device (torch.device): Target device (e.g., torch.device('cuda')).

    Returns
    -------
        torch.Tensor: Complex tensor of shape (n_ch, n_pol, n_s) representing
                      the laser waveform.
    """
    # Initiate the seed of random numbers
    manual_seed(seed)

    # discrete time index
    n = arange(0, n_s, dtype=float32, device=device)

    # Sampling frequency
    fs = sr * k

    # Calculating the linear power of the continuous wave:
    pcw_linear = tensor(10 ** ((power_dbm - 30) / 10))

    # Generating the electric field of the optical signal
    E = ones((n_ch, n_pol, n_s), dtype=float32, device=device) * \
        sqrt(pcw_linear / n_pol)

    # If the linewidth is greater than 0 Hz
    if lw > 0:

        # Period between samples at the (oversampled) transmitted signal:
        T = 1 / (k * sr)

        # Calculating the phase noise:
        var = tensor(2 * pi * lw * T)
        delta_theta = sqrt(var) * randn((n_ch, 1, n_s), dtype=float32,
                                        device=device)
        theta = cumsum(delta_theta, dim=-1)

        # Phase noise
        phase_noise = exp(1j * theta)

        # Adding phase noise to the optical signal:
        E = E * phase_noise.repeat(1, n_pol, 1)

    return E * exp(1j * 2 * pi * f_grid.view(-1, 1, 1) * n.view(1, 1, -1) / fs)
