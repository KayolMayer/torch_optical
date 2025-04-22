"""
Created on Tue Apr 22 14:36:37 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from packages.opt_tx import laser_tx
from torch import tensor, cfloat, sqrt, pi, zeros_like, stack, exp
# =============================================================================
# =============================================================================


def laser_rx(n_ch, n_pol, n_s, power_dbm, sr, k, lw, f_grid, seed, device):
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
    E = laser_tx(n_ch, n_pol, n_s, power_dbm, sr, k, lw, f_grid, seed, device)

    return E


def optical_front_end(Er, ELo, responsivity, phase_shift, device):
    """
    Simulate a coherent optical front-end.

    It simulates 90-degree hybrid and balanced photodetectors.

    Args
    ----
        Er (torch.Tensor): Received optical signal (complex),
                           shape (n_ch, n_pol, n_samples).
        ELo (torch.Tensor): Local oscillator signal, same shape as Er.
        responsivity (float): Photodiode responsivity (A/W).
        phase_shift (float): Phase offset between LO and signal (radians).
        device (torch.device): Device to run computations on.

    Returns
    -------
        torch.Tensor: Complex-valued electrical signal after optical front-end,
                      shape (n_ch, n_pol, n_samples), where:
                        - Real part = I (in-phase)
                        - Imag part = Q (quadrature)
    """
    # 90-degree optical hybrid splits the inputs into 4 outputs
    E1, E2, E3, E4 = __hybrid90(Er, ELo, phase_shift, device)

    # Photodetection: intensity = responsivity × |E|²
    # In-phase
    i1 = responsivity * (E1 * E1.conj())
    i2 = responsivity * (E2 * E2.conj())

    # Quadrature
    i3 = responsivity * (E3 * E3.conj())
    i4 = responsivity * (E4 * E4.conj())

    # Balanced detection (subtract pairs to remove common-mode noise) (V and H)
    iVHI = i1 - i2
    iVHQ = i3 - i4

    return iVHI + 1j*iVHQ


def __hybrid90(Er, ELo, phase_shift, device):
    """
    Implement a 90-degree optical hybrid using 3-dB couplers in PyTorch.

    Args
    ----
        Er (torch.Tensor): Received signal, shape (n_ch, n_pol, n_samples).
        ELo (torch.Tensor): Local oscillator signal, same shape as Er.
        phase_shift (float): Additional phase shift (degrees) applied to one
                             LO branch.
        device (torch.device): Target device for computation.

    Returns
    -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            E1, E2, E3, E4 — 4 output fields from the hybrid, each of
            shape (n_ch, n_pol, n_samples)
    """
    # 3-dB coupler transfer function:
    Hc = (1 / sqrt(tensor(2, device=device))) * \
        tensor([[1, 1], [1, -1]], dtype=cfloat, device=device)

    # Convert the phase shift for a tensor
    phase_shift = tensor(phase_shift, dtype=cfloat, device=device)

    # Convert phase_shift from degrees to radians and apply to phase term
    phase = exp(1j * (1 + (phase_shift * pi / 180)) * pi / 2)

    # Repeat Er to have the same dimension of the laser for the channels
    Er = Er.repeat(ELo.shape[0], 1, 1)

    # ECouplerTL - Signal at the output of the top-left 3-dB coupler at the
    # 90 degree hybrid
    ECouplerTL = Hc @ stack((Er, zeros_like(Er)), dim=-2)

    # ECouplerBL - Signal at the output of the bottom-left 3-dB coupler at
    # the 90 degree hybrid
    ECouplerBL = Hc @ stack((ELo, zeros_like(ELo)), dim=-2)

    # ECouplerTR - Signal at the output of the top-right 3-dB coupler at
    # the 90 degree hybrid
    ECouplerTR = Hc @ stack((ECouplerTL[..., 0, :], ECouplerBL[..., 0, :]),
                            dim=-2)

    # ECouplerBR - Signal at the output of the bottom-right 3-dB coupler at
    # the 90 degree hybrid
    ECouplerBR = Hc @ stack((ECouplerTL[..., 1, :],
                             ECouplerBL[..., 1, :] * phase), dim=-2)

    # Output signals E1, E2, E3 and E4
    return (
        ECouplerTR[..., 0, :],
        ECouplerTR[..., 1, :],
        ECouplerBR[..., 0, :],
        ECouplerBR[..., 1, :]
    )
