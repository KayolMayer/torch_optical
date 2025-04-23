"""
Created on Tue Apr 22 14:36:37 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from packages.opt_tx import laser_tx
from torch import tensor, cfloat, float32, sqrt, pi, zeros_like, stack, exp, \
    arange, cat
from torch.nn.functional import grid_sample
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


def insert_skew(signal, k, sr, par_skew, device):
    """
    Apply independent skew (delay or advance) to I and Q of each polarization.

    Args
    ----
        signal (torch.Tensor): Complex tensor of shape (n_ch, n_pol, n_samples)
        k (int): Oversampling factor
        sr (float): Symbol rate in Hz
        par_skew (list or tensor): Length = 2 * n_pol, skew values in seconds.
        device (torch.device): Device to run computations on.

    Returns
    -------
        torch.Tensor: Tensor of shape (n_ch, n_pol, n_samples_trimmed)
    """
    n_ch, n_pol, n_samples = signal.shape

    # Convert skew to tensor in sample units
    par_skew = tensor(par_skew, dtype=float32, device=device)
    skew_samples = (par_skew - par_skew.min()) * k * sr  # in sample units

    signal_out_real = []
    signal_out_imag = []

    # Apply skew per polarization
    for pol in range(n_pol):
        real_part = signal[:, pol, :].real
        imag_part = signal[:, pol, :].imag

        skew_I = skew_samples[2 * pol]
        skew_Q = skew_samples[2 * pol + 1]

        grid_pos_real = arange(n_samples, device=device).float() - skew_I
        grid_pos_imag = arange(n_samples, device=device).float() - skew_Q

        shifted_real = __interpolate_1d(real_part, grid_pos_real, device)
        shifted_imag = __interpolate_1d(imag_part, grid_pos_imag, device)

        signal_out_real.append(shifted_real.unsqueeze(1))  # (n_ch, 1, L)
        signal_out_imag.append(shifted_imag.unsqueeze(1))

    # Stack and reconstruct complex signal
    real_combined = cat(signal_out_real, dim=1)  # (n_ch, n_pol, L)
    imag_combined = cat(signal_out_imag, dim=1)  # (n_ch, n_pol, L)
    output = real_combined + 1j * imag_combined

    return output


def __interpolate_1d(x, grid_pos, device):
    """
    Apply 1D delay/advance using bilinear interpolation via grid_sample.

    Args
    ----
        x (torch.Tensor): (n_ch, n_samples), real-valued
        grid_pos (torch.Tensor): grid_position for interpolation

    Returns
    -------
        torch.Tensor: (n_ch, n_samples), interpolated signal
    """
    n_ch, L = x.shape
    x = x.unsqueeze(1).unsqueeze(2)  # (n_ch, 1, 1, L) → [N, C, H=1, W=L]

    # Create normalized 1D grid: (N, H=1, W=L, 2)
    grid_x = grid_pos / (L - 1) * 2 - 1  # Normalize to [-1, 1]
    grid_y = zeros_like(grid_x)    # y = 0 for 1D

    # Combine and reshape grid to (N, 1, L, 2)
    grid = stack((grid_x, grid_y), dim=-1)  # (L, 2)
    # (n_ch, 1, L, 2)
    grid = grid.unsqueeze(0).expand(n_ch, 1, L, 2).contiguous()

    # grid_sample expects input shape [N, C, H, W], grid shape [N, H, W, 2]
    out = grid_sample(x, grid, mode='bilinear', padding_mode='border',
                      align_corners=True)

    return out.squeeze(2).squeeze(1)  # Back to shape (n_ch, L)


def adc(signal, k, samples, f_error, p_error, device):
    """
    Simulate the ADC, including phase error and frequency offset correction.

    Args
    ----
        signal (torch.Tensor): Input complex signal of shape (n_ch, n_pol,
                                                              n_samples),
                               where n_ch is number of WDM channels, and n_pol
                               is the number of polarizations.
        k (int): Oversampling factor (samples per symbol).
        samples (int): Number of samples per symbol after downsampling
                       (typically 1 or 2).
        f_error (float): Frequency offset error (Hz).
        p_error (float): Phase offset error (fraction of a sample).
        device (torch.device): Target device for computations (e.g., 'cuda').

    Returns
    -------
        torch.Tensor: Output complex signal after ADC and downsampling,
                      shape (n_ch, n_pol, n_output_samples).
    """
    n_ch, n_pol, n_samples = signal.shape

    # Sampling positions (same for all polarizations initially)
    grid_pos_base = arange(n_samples, dtype=float32, device=device) +\
        (k * f_error / samples)

    signal_out_real = []
    signal_out_imag = []

    # Apply skew per polarization
    for pol in range(n_pol):
        real_part = signal[:, pol, :].real
        imag_part = signal[:, pol, :].imag

        # Total sampling positions with phase error
        grid_pos = grid_pos_base - p_error * k

        shifted_real = __interpolate_1d(real_part, grid_pos, device)
        shifted_imag = __interpolate_1d(imag_part, grid_pos, device)

        signal_out_real.append(shifted_real.unsqueeze(1))  # (n_ch, 1, L)
        signal_out_imag.append(shifted_imag.unsqueeze(1))

    # Stack and reconstruct complex signal
    real_combined = cat(signal_out_real, dim=1)  # (n_ch, n_pol, L)
    imag_combined = cat(signal_out_imag, dim=1)  # (n_ch, n_pol, L)
    output = real_combined + 1j * imag_combined

    # Downsample
    step = int(k / samples)
    output = output[..., ::step]

    return output
