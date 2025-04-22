"""
Created on Tue Apr 22 11:30:53 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import manual_seed, float32, arange, sqrt, ones, pi, tensor, exp, \
    randn, cumsum, cos, amax
from torch import abs as abs_torch
from torch import sum as sum_torch
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
    pcw_linear = tensor(10 ** ((power_dbm - 30) / 10), dtype=float32,
                        device=device)

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

    # Apply frequency shift per channel (frequency grid)
    phase_c = exp(1j * 2 * pi * f_grid.view(-1, 1, 1) * n / fs)
    E = E * phase_c

    return E


def iqModulator(qam_tensor, laser, maxExc, minExc, bias, vpi):
    """
    Simulate an IQ optical modulator.

    It applies a bias and phase modulation to the in-phase and quadrature
    components of the input electrical signals.

    Args
    ----
        qam_tensor (torch.Tensor): Complex-valued input modulation signal of
                                   shape (n_ch, n_pol, n_samples).
        laser (torch.Tensor): Optical carrier signal (same shape as
                              qam_tensor), complex-valued.
        maxExc (float): Maximum excursion voltage of the modulator.
        minExc (float): Minimum excursion voltage of the modulator.
        bias (float): Bias voltage applied to each arm of the modulator.
        vpi (float): V_pi voltage of the modulator (phase shift of π).

    Returns
    -------
        torch.Tensor: Modulated optical signal (same shape), complex-valued.
    """
    # Obtaining the in-phase and quadrature components of the electrical IQs
    mI = qam_tensor.real
    mQ = qam_tensor.imag

    # Normalize I and Q independently (per channel/polarization)
    mI = mI / amax(abs_torch(mI), dim=-1, keepdim=True)
    mQ = mQ / amax(abs_torch(mQ), dim=-1, keepdim=True)

    # Setting the signal excursion:
    mI = mI * (maxExc - minExc) / 2
    mQ = mQ * (maxExc - minExc) / 2

    # Obtaining the signals after considering the bias:
    vI = mI + bias
    vQ = mQ + bias

    # Phase modulation in the in-phase and quadrature branches
    phiI = pi * vI / vpi
    phiQ = pi * vQ / vpi

    # IQM output signal:
    signal_out = (0.5 * cos(0.5 * phiI) + 0.5j * cos(0.5 * phiQ)) * laser

    return signal_out


def mux(signals):
    """
    Multiplex a multi-channel signal by summing over the channel dimension.

    Args
    ----
        signals (torch.Tensor): Input tensor of shape (n_ch, n_pol, n_samples),
                                typically complex-valued.

    Returns
    -------
        torch.Tensor: Summed (muxed) signal of shape (n_pol, n_samples).
    """
    return sum_torch(signals, dim=0, keepdim=True)
