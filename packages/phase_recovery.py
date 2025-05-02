"""
Created on Fri May  2 10:55:51 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import pi, arange, float32, zeros, exp, cat, argmin, floor
from torch import sum as sum_torch
from torch import abs as abs_torch
from packages.data_streams import quantization_qam, denorm_qam_power, \
    norm_qam_power
# =============================================================================
# =============================================================================


def phase_recovery_bps(signal, qam_order, n_symb, n_test, device):
    """
    Perform phase recovery using the Blind Phase Search (BPS) algorithm.

    This method evaluates multiple candidate carrier phases for each symbol,
    applies hard decisions to each rotated signal, and selects the phase that
    minimizes the mean squared error between the rotated signal and its nearest
    constellation point. It then unwraps the estimated phase trajectory and
    compensates the input signal accordingly.

    Args
    -----
        signal (torch.Tensor): Input complex signal,
                               shape (n_ch, n_pol, n_samples)
        qam_order (int): QAM modulation order (e.g., 4, 16, 64)
        n_symb (int): Half window size for error averaging
                      (total window = 2*n_symb+1)
        n_test (int): Number of test phase angles (uniformly spaced over π/2)
        device (torch.device): PyTorch device to run on (e.g., "cuda")

    Returns
    -------
        torch.Tensor: Phase-corrected signal, same shape as input

    Notes
    -----
        - Assumes square QAM with π/2 rotational symmetry (i.e., 4-fold).
        - Applies sliding window averaging to smooth error estimates.
        - Performs phase unwrapping to ensure continuity in the estimated
          phase.
        - Relies on a hard-decision function to quantize symbols.
    """
    # Normalize the input signal
    signal = norm_qam_power(signal, 'power', qam_order)

    # Get the number of channels, polarizations, and samples
    n_ch, n_pol, n_s = signal.shape

    # Creating a vector of test carrier phase angles:
    theta_test = pi / 2 * arange(-n_test/2, n_test/2, dtype=float32,
                                 device=device) / n_test

    # Creating a vector of test phase rotations
    phase_test = exp(-1j * theta_test)

    # Apply all the phase tests
    sig_rot = signal.unsqueeze(-1) * phase_test

    # Apply the hard decision operation for all phase tests
    sig_rot_dec = __hard_decision(sig_rot, qam_order, device)

    # Get the distance between the phase rotations and the hard decision
    errors = abs_torch(sig_rot - sig_rot_dec) ** 2

    # Concatenate zeros to perform the summation correctly in the boundaries
    zeros_t = zeros((n_ch, n_pol, n_symb, n_test), dtype=float32,
                    device=device)
    errors = cat((zeros_t, errors.real, zeros_t), dim=-2)

    # Initialize the error sum tensor
    e_sum = zeros((n_ch, n_pol, n_s, n_test), dtype=float32, device=device)

    for ii in range(n_s):

        # Compute the errors sum
        e_sum[..., ii, :] = sum_torch(errors[..., ii-n_symb: ii+n_symb+1, :],
                                      dim=-2)

    # Get the angle that generated the minimum error
    theta_min = theta_test[argmin(e_sum, dim=-1)]

    # Concatenate zeros to perform the unwrapping correctly
    zeros_t = zeros((n_ch, n_pol, 1), dtype=float32, device=device)
    theta_min = cat((zeros_t, theta_min), dim=-1)

    # Apply the phase unwrapping
    for ii in range(1, n_s + 1):

        # Compute the phase unwrapping
        theta_min[..., ii] = theta_min[..., ii] + \
            floor(0.5 - (theta_min[..., ii] -
                         theta_min[..., ii-1]) / (pi / 2)) * pi / 2

    # Discard the initial phase added to apply the unwrapping correctly
    theta_min = theta_min[..., 1:]

    # Compensate the phase noise
    out = signal * exp(-1j * theta_min)

    return out


def __hard_decision(signal, qam_order, device):
    """
    Apply hard-decision quantization to a QAM signal.

    The input signal is first denormalized to match the native power of the
    QAM constellation, then quantized to the nearest valid QAM symbol, and
    finally normalized back (if required) for consistent output power.

    Args
    ----
        signal (torch.Tensor): Input signal (complex-valued), shape (...,)
        qam_order (int): QAM modulation order (must be a square, e.g., 16, 64)
        device (torch.device): Device on which to perform computation

    Returns
    -------
        torch.Tensor: Hard-decision quantized signal (same shape as input)

    Notes
    -----
        - This function supports square QAM constellations only.
        - Uses minimum-distance quantization via 'quantization_qam()'.
        - Normalization uses the known analytical average QAM power.
    """
    # Denormalize symbols to the square QAM power
    sig_denorm = denorm_qam_power(signal, qam_order)

    # Quantize symbols
    sig_quant = quantization_qam(sig_denorm, qam_order, device)

    # Normalize symbols to unitary power
    out = norm_qam_power(sig_quant, 'constellation', qam_order)

    return out
