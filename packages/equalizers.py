"""
Created on Fri Apr 25 13:35:51 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import tensor, ceil, cat, reshape, zeros, cfloat, float32, arange, \
    exp, pi, roll, meshgrid, mean, sqrt, flip
from torch import sum as sum_torch
from torch import abs as abs_torch
from torch.fft import fft, fftshift, ifft, ifftshift
# =============================================================================
# =============================================================================


def cd_equalization(signal, disp, fib_len, freq, sr, k, n_fft, n_over, device):
    """
    Apply chromatic dispersion (CD) equalization to an optical signal.

    Chromatic dispersion is compensated by applying the inverse of the fiber's
    CD transfer function, assuming known dispersion and fiber length.

    Args
    ----
        signal (torch.Tensor): Input signal of shape (n_ch, n_pol, n_samples)
        disp (float): Dispersion coefficient [ps/nm/km]
        fib_len (float): Fiber length [km]
        freq (torch.Tensor): Optical carrier frequencies [Hz], shape (n_ch,)
        sr (float): Symbol rate [Hz]
        k (int): Oversampling factor (samples per symbol)
        n_fft (int): FFT size used in overlap-save
        n_over (int): Number of overlapping samples between FFT windows
        device (torch.device): Device for computation (e.g., "cuda")

    Returns
    -------
        torch.Tensor: Dispersion-equalized output signal, shape same as input

    Notes
    -----
        - The function compensates only for CD (not PMD or nonlinearity).
        - Uses a frequency-domain kernel based on quadratic phase rotation.
        - Dispersion parameter is converted from [ps/nm/km] to [s/m^2].
        - Applies overlap-save block processing for long sequences.
    """
    # Speed of light [m/s]
    c = 299792458

    # Center lambda [m]
    c_lambda = (c / freq).view(-1, 1, 1)

    # fiber length in meters
    L = fib_len * 1e3

    # Dispersion parameter [ps/m^2]
    disp = disp * 1e-6

    # Index for coefficient calculation
    n = arange(-n_fft / 2, n_fft / 2, 1, dtype=float32, device=device)

    # Nyquist frequency
    f_n = k * sr / 2

    # Calculating the CD frequency response
    h_freq_cd = exp(-1j * pi * c_lambda ** 2 * disp * L / c *
                    (2 * n * f_n / n_fft) ** 2).unsqueeze(-2)

    # Guaranteeing that nOverlap is even
    n_over = n_over + (n_over % 2)

    out = __overlap_cdc(signal, n_fft, n_over, h_freq_cd, device)

    return out


def __overlap_cdc(signal, n_fft, n_over, h_freq_cd, device):
    """
    Apply overlap-save based frequency-domain convolution..

    This function processes each channel and polarization separately using
    windowed blocks with overlap, applying the filter in the frequency domain
    (FDE - Frequency Domain Equalization).

    Args
    ----
        signal (torch.Tensor): Input signal of shape (n_ch, n_pol, n_samples)
        n_fft (int): FFT size (must be a power of 2 for best performance)
        n_over (int): Number of overlapping samples between FFT blocks
                      even number
        h_freq_cd (torch.Tensor): Frequency response of the CD compensation
                                  filter, shape (n_ch, 1, n_fft)
        device (torch.device): Target device for PyTorch computation

    Returns
    -------
        torch.Tensor: Filtered output signal of shape (n_ch, n_pol, n_samples)

    Notes
    -----
        - Overlap is symmetrically padded on both sides of the input signal.
        - Blocks are assembled into 'n_fft'-length segments for FFT processing.
        - Output signal length is trimmed to match the original input.
        - 'fftshift' and 'ifftshift' are used for frequency alignment.
        - Assumes input 'signal' is complex-valued (dtype=cfloat).
    """
    # Get the number of channels, polarizations, and samples
    n_ch, n_pol, n_s = signal.shape

    # Extending the input signal so that the blocks can be properly formed
    aux_len = tensor(n_s / (n_fft - n_over), device=device)

    if aux_len != ceil(aux_len):
        n_extra = ceil(aux_len) * (n_fft - n_over) - n_s
    else:
        n_extra = n_over

    # Extended input signals
    sig_in = cat((signal[..., -int(n_extra/2):],
                  signal,
                  signal[..., :int(n_extra/2)]), dim=-1)

    # Get the number of extended samples
    n_s_e = sig_in.shape[-1]

    # Number of blocks
    n_b = int(n_s_e / (n_fft - n_over))

    # Blocks
    blocks = reshape(sig_in, (n_ch,
                              n_pol,
                              n_b,
                              int(n_fft - n_over)))

    # Preallocating the block overlaps
    overlap = zeros((n_ch, n_pol, n_b, n_fft), dtype=cfloat, device=device)

    overlap[..., 0, :] = cat((overlap[..., 0, -int(n_over):],
                              blocks[..., 0, :]), dim=-1)
    for ii in range(1, n_b):
        overlap[..., ii, :] = cat((overlap[..., ii-1, -int(n_over):],
                                   blocks[..., ii, :]), dim=-1)

    # FFT of the overlaped blocks
    overlap_freq = fftshift(fft(overlap))

    # Filtering in frequency domain
    out_fde_freq = overlap_freq * h_freq_cd

    # IFFT of the block after filtering
    out_fde = ifft(ifftshift(out_fde_freq))

    # Output block
    out_blocks = out_fde[..., int(n_over / 2): -int(n_over / 2)]

    # Take out the block dimension
    out = reshape(out_blocks, (n_ch, n_pol, n_s_e))

    # Discard initial overlap
    out = out[..., int((n_extra + n_over) / 2): -int((n_extra - n_over) / 2)]

    return out


def cma_equalization(signal, k, taps, eta, qam_order, norm, device, spike=0,
                     n_iter_sing=5000):
    """
    Perform Constant Modulus Algorithm (CMA) equalization.

    This implementation processes complex signals received over two
    polarizations (V and H) and adaptively updates four FIR filters using the
    CMA cost function.

    Args
    ----
        signal (torch.Tensor): Input signal of shape (n_ch, 2, n_samples)
        k (int): Oversampling factor (samples per symbol)
        taps (int): Number of taps in each FIR filter
        eta (float): CMA step size (learning rate)
        qam_order (int): QAM modulation order (e.g., 16, 64)
        norm (bool): If True, constellation is normalized to unit power
        device (torch.device): PyTorch device (e.g., 'cuda' or 'cpu')
        spike (int): Index of tap to initialize as a delta (1 + 0j), default=0
        n_iter_sing (int): Iteration to trigger singularity reset on y₂ filters

    Returns
    -------
        torch.Tensor: Equalized signal, shape (n_ch, 2, n_symbols)

    Notes
    -----
        - The CMA equalizer minimizes the deviation from a constant modulus.
        - This version assumes two polarizations and uses 2×2 adaptive
          filtering.
        - Filter tap matrices:
              y₀ = w₀ᵥ * xᵥ + w₀ₕ * xₕ
              y₁ = w₁ᵥ * xᵥ + w₁ₕ * xₕ
        - After 'n_iter_sing' steps, the second output filters are
          reinitialized from the first to avoid degenerate convergence.
    """
    # Get the number of channels, polarizations, and samples
    n_ch, n_pol, n_s = signal.shape

    # Number ofsymbolsafter equalization
    n_s_e = int(n_s / k)

    # Compute the CMA radius of the contellations
    r = __square_qam_cma_radius(qam_order, norm)

    # Create filters
    w_0_v = zeros((n_ch, taps), dtype=cfloat, device=device)
    w_1_v = zeros((n_ch, taps), dtype=cfloat, device=device)
    w_0_h = zeros((n_ch, taps), dtype=cfloat, device=device)
    w_1_h = zeros((n_ch, taps), dtype=cfloat, device=device)

    # Initialize the vertical channel (regarding y_1)
    w_0_v[:, spike] += 1 + 0j

    # Create FIFOs
    fifo = zeros((n_ch, n_pol, taps), dtype=cfloat, device=device)

    # Create outputs
    y = zeros((n_ch, n_pol, n_s_e), dtype=cfloat, device=device)

    for ii in range(0, n_s_e):

        # Insert the first element in the FIFO
        fifo = roll(fifo, shifts=k, dims=-1)
        fifo[..., 0] = signal[..., int(ii * k)]
        fifo[..., 1] = signal[..., int(ii * k + 1)]

        # Update the outputs
        y[:, 0, ii] = sum_torch(w_0_v * fifo[:, 0, :] +
                                w_0_h * fifo[:, 1, :], dim=-1)

        y[:, 1, ii] = sum_torch(w_1_v * fifo[:, 0, :] +
                                w_1_h * fifo[:, 1, :], dim=-1)

        # auxiliar computation
        aux = (r - abs_torch(y[..., ii])**2) * y[..., ii]

        # Update the filters
        w_0_v += eta * fifo[:, 0, :].conj() * aux[:, 0].unsqueeze(-1)

        w_0_h += eta * fifo[:, 1, :].conj() * aux[:, 0].unsqueeze(-1)

        w_1_v += eta * fifo[:, 0, :].conj() * aux[:, 1].unsqueeze(-1)

        w_1_h += eta * fifo[:, 1, :].conj() * aux[:, 1].unsqueeze(-1)

        # Reinitialize the filters of y2 to avoid singularities
        if ii == n_iter_sing:
            w_1_h = flip(w_0_v, dims=(-1,)).conj()
            w_1_v = -flip(w_0_h, dims=(-1,)).conj()

    return y


def __square_qam_cma_radius(qam_order, normalized=True):
    """
    Compute the ideal CMA radius for a square QAM constellation.

    The CMA radius is defined as:
        R = E[|x|^4] / E[|x|^2]
    where x are the QAM constellation points. This radius is used in the CMA
    error function to guide adaptive equalization.

    Args
    ----
        qam_order (int): QAM modulation order (e.g., 4, 16, 64, 256)
        normalized (bool): If True, normalize constellation to unit average
        power

    Returns
    -------
        float: The ideal CMA radius for the specified QAM order
    """
    levels = int(qam_order ** 0.5)
    axis = arange(levels, dtype=float32)
    axis = 2 * axis - (levels - 1)  # Center constellation at 0

    # Meshgrid over I and Q
    I, Q = meshgrid(axis, axis, indexing='ij')
    constellation = I + 1j * Q

    # Flatten to 1D tensor
    symbols = constellation.reshape(-1)

    if normalized:
        power = mean(abs_torch(symbols) ** 2)
        symbols = symbols / sqrt(power)

    # Compute the radius
    radius = mean(abs_torch(symbols) ** 4) / mean(abs_torch(symbols) ** 2)

    return radius
