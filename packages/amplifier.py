"""
Created on Thu Apr 24 12:43:26 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import log, exp, manual_seed, sqrt, tensor, float32, randn
# =============================================================================
# =============================================================================


def __gain_edfa(signal, gain_db, device):
    """
    Apply EDFA (Erbium-Doped Fiber Amplifier) gain to a complex signal.

    The function amplifies the input signal using a specified gain in dB.
    The signal is scaled in the complex domain by the square root of the
    linear gain to model amplitude amplification.

    Args
    ----
        signal (torch.Tensor): Complex input tensor of
                               shape (n_ch, n_pol, n_samples)
        gain_db (float or torch.Tensor): Gain in decibels (dB)
        device (torch.device): Target device (e.g., torch.device('cuda')).

    Returns
    -------
        torch.Tensor: Amplified signal of the same shape as input
    """
    # Linear gain
    gain_lin = gain_db * log(tensor(10, dtype=float32, device=device)) / 10

    # Signal after amplification
    out_data = signal * exp(gain_lin / 2)

    return out_data


def __ase_edfa(signal, nf_db, gain_db, freq, sr, alpha, k_up, seed, device):
    """
    Add ASE (Amplified Spontaneous Emission) noise to an optical signal.

    This function models the noise introduced by an Erbium-Doped Fiber
    Amplifier (EDFA) using the standard quantum-limited ASE formulation.
    Gaussian noise is added to the input signal with a power determined by the
    gain, noise figure, and system parameters.

    Args
    ----
        signal (torch.Tensor): Complex input tensor of
                               shape (n_ch, n_pol, n_samples)
        nf_db (float): EDFA noise figure in dB
        gain_db (float): Amplifier gain in dB
        freq (torch.Tensor): Tensor of frequencies (n_ch)
        sr (float): Symbol rate in Hz
        alpha (float): Filter rolloff
        k_up (int): Upsampling factor (samples per symbol)
        seed (int): Seed for random number generation (for reproducibility)
        device (torch.device): Target device (e.g., torch.device("cuda"))

    Returns
    -------
        torch.Tensor: Output signal with ASE noise, same shape as input
    """
    # Initiate the seed of random numbers
    manual_seed(seed + 2)

    # Planck's constant [m^2 kg / s]
    h = 6.62607004e-34

    # Number of channels, polarizations, and samples
    n_ch, n_pol, n_s = signal.shape

    # Linear noise figure
    f_lin = 10 ** (nf_db/10)

    # Linear gain
    gain_lin = tensor(10 ** (gain_db / 10), dtype=float32, device=device)

    # ASE noise power
    p_ase = f_lin * h * freq * sr * alpha * gain_lin

    # Standard deviation of noise
    std_dev = sqrt(p_ase * k_up * n_pol / 4)

    noise = randn((n_ch, n_pol, n_s), dtype=float32, device=device) + \
        1j * randn((n_ch, n_pol, n_s), dtype=float32, device=device)

    out_data = signal + std_dev.view(-1, 1, 1) * noise

    return out_data


def edfa(signal, nf_db, gain_db, freq, sr, alpha, k_up, seed, device):
    """
    Simulate the effect of an EDFA on an optical signal.

    This function applies both gain and Amplified Spontaneous Emission (ASE)
    noise to the input signal, modeling a realistic EDFA component in optical
    systems.

    Args
    ----
        signal (torch.Tensor): Complex input signal of
                               shape (n_ch, n_pol, n_samples)
        nf_db (float): Noise figure of the EDFA in dB
        gain_db (float): Gain of the amplifier in dB
        freq (torch.Tensor): Tensor of frequencies (n_ch)
        sr (float): Symbol rate in Hz
        alpha (float): Filter rolloff
        k_up (int): Upsampling factor (samples per symbol)
        seed (int): Random seed for reproducible ASE noise
        device (torch.device): Target device (e.g., torch.device("cuda"))

    Returns
    -------
        torch.Tensor: Amplified signal with ASE noise, same shape as input
    """
    # Edfa gain
    signal = __gain_edfa(signal, gain_db, device)

    # Edfa ase
    out_data = __ase_edfa(signal, nf_db, gain_db, freq, sr, alpha, k_up, seed,
                          device)

    return out_data
