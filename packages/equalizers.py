"""
Created on Fri Apr 25 13:35:51 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import tensor, ceil, cat, reshape, zeros, cfloat, float32, arange, \
    exp, pi
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
