"""
Created on Tue April 15 16:07:28 2025.

@author: Kayol Mayer
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import manual_seed, randint, reshape, arange, int8, cat, tensor, \
    long, sqrt, cumsum, roll, mean, argmin, float32, meshgrid, log2, zeros, \
    flip, cfloat, pi, exp
from torch import sum as sum_torch
from torch import abs as abs_torch
from torch import max as max_torch
from torch import round as round_torch
from packages.utils import p_corr
# =============================================================================
# =============================================================================


def random_square_qam_sequence(n_ch, n_pol, n_symb, qam_order, device, seed,
                               gray=True, norm=True):
    """
    Generate a random sequence of square QAM symbols for simulations.

    This function creates a random bit sequence and maps it to a complex-valued
    QAM symbol tensor using Gray coding and optional normalization to unit
    power.

    Args
    ----
        n_ch (int): Number of channels.
        n_pol (int): Number of polarizations (1 or 2).
        n_symb (int): Number of QAM symbols to generate per channel/pol.
        qam_order (int): The QAM modulation order (e.g., 4, 16, 64, 256).
        device (torch.device): The device (CPU or GPU) to perform computations.
        seed (int): Random seed for reproducibility.
        gray (bool, optional): Whether to apply Gray coding to bit-to-symbol
                               mapping. Default is True.
        norm (bool, optional): Whether to normalize the QAM constellation to
                               unit average power. Default is True.

    Returns
    -------
        torch.Tensor: A complex-valued tensor of shape (n_ch, n_pol, n_symb),
                      containing the QAM symbols (dtype: complex64 or
                                                  complex128 depending on
                                                  backend).

    Example
    -------
        >>> device = torch.device("cpu")
        >>> symbols = random_square_qam_sequence(n_ch=2, n_pol=2, n_symb=100,
                                                 qam_order=16, device=device,
                                                 seed=42)
        >>> print(symbols.shape)
        torch.Size([2, 2, 100])
    """
    # Ensure square QAM, get bits per I/Q component, and power for norm
    bits_qam, _, _ = __square_qam_param(qam_order)

    bit_tensor = random_bit_sequence(n_ch, n_pol, bits_qam * n_symb, device,
                                     seed)

    symb_tensor = bit_to_qam(bit_tensor, qam_order, device, gray=gray,
                             norm=norm)

    return symb_tensor


def random_bit_sequence(n_ch, n_pol, n_bits, device, seed):
    """
    Generate a random bit sequence as a tensor.

    This function creates a tensor of random binary values (0 or 1)
    with dimensions corresponding to the number of channels, polarizations,
    and bits. The output is generated using a fixed random seed for
    reproducibility.

    Args
    ----
        n_ch (int): Number of channels.
        n_pol (int): Number of polarizations (must be 1 or 2).
        n_bits (int): Number of bits per channel and polarization.
        device (torch.device): The device (CPU or GPU) on which to create the
                               tensor.
        seed (int): Seed for the random number generator (for reproducibility).

    Returns
    -------
        torch.Tensor: A tensor of shape (n_ch, n_pol, n_bits) with random bits
                     (0 or 1).

    Raises
    ------
        ValueError: If 'n_pol' is not 1 or 2.

    Example
    -------
        >>> device = torch.device("cpu")
        >>> bits = random_bit_sequence(n_ch=4, n_pol=2, n_bits=100,
                                       device=device, seed=42)
        >>> bits.shape
        torch.Size([4, 2, 100])
    """
    manual_seed(seed)

    # Verify if the number of polarizations is ok
    if n_pol != 1 and n_pol != 2:
        raise ValueError("Number of polarizations must be 1 or 2!")

    bit_tensor = randint(2, size=(n_ch, n_pol, n_bits), device=device)

    return bit_tensor


def bit_to_qam(bit_tensor, qam_order, device, gray=True, norm=True):
    """
    Convert a tensor of binary bits to a QAM-modulated complex symbol sequence.

    This function supports various square QAM modulations (e.g., 16-QAM),
    with optional Gray coding and constellation normalization to unit average
    power. If n_bits divided by the number of bits per QAM symbol is not
    an integer, n_bits is reduced to make it matches.

    Args
    ----
        bit_tensor (torch.Tensor): A 3D tensor of binary bits with shape
                                   (n_ch, n_pol, n_bits).
        qam_order (int): The modulation order (e.g., 4, 16, 64, 256, etc.).
        device (torch.device): The device (CPU or GPU) for computations.
        gray (bool, optional): Whether to apply Gray coding before modulation.
                               Default is True.
        norm (bool, optional): Whether to normalize constellation to unit
                               average power. Default is True.

    Returns
    -------
        torch.Tensor: A tensor of complex QAM symbols with shape
                      (n_ch, n_pol, n_symb), dtype 'torch.complex64'.

    Example
    -------
        >>> bits = torch.randint(0, 2, (2, 2, 240), dtype=torch.int8)
        >>> qam = bit_to_qam(bits, qam_order=16, device=torch.device("cpu"))
        >>> print(qam.shape)
        torch.Size([2, 2, 60])
    """
    # Ensure square QAM, get bits per I/Q component, and power for norm
    n_bits, levels, power = __square_qam_param(qam_order)

    # Number of bit blocks to fit the number of symbols
    n_blocks = int(bit_tensor.size(dim=2)//n_bits)

    # Bit tensor after clipping extra bits
    bit_tensor = bit_tensor[:, :, 0:n_blocks * n_bits]

    # Split the bit sequence into blocks for I/Q symbol generation
    bit_blocks = reshape(bit_tensor, (bit_tensor.size(dim=0),
                                      bit_tensor.size(dim=1),
                                      n_blocks, n_bits))

    if gray:
        # Convert from gray to binary
        binary_i = __gray_to_binary(bit_blocks[:, :, :, :int(n_bits/2)])
        binary_q = __gray_to_binary(bit_blocks[:, :, :, int(n_bits/2):])

        # Convert from binary to decimal
        decimal_i = __binary_to_decimal(binary_i, device)
        decimal_q = __binary_to_decimal(binary_q, device)

    else:

        # Convert from binary to decimal
        decimal_i = __binary_to_decimal(bit_blocks[:, :, :, :int(n_bits/2)],
                                        device)
        decimal_q = __binary_to_decimal(bit_blocks[:, :, :, int(n_bits/2):],
                                        device)

    # Convert from decimal to QAM
    qam = (decimal_i * 2 - (levels - 1)) + 1j * (decimal_q * 2 - (levels - 1))

    if norm:
        # Normalization by the constellation power
        qam = qam / sqrt(tensor(power))

    return qam


def qam_to_bit(qam_tensor, qam_order, device, gray=True):
    """
    Convert square QAM symbols into their corresponding binary bit sequences.

    Supports standard square QAM modulations (e.g., 4-QAM, 16-QAM, etc.) with
    optional Gray decoding. Returns the binary bit sequence corresponding to
    each QAM symbol.

    Args
    ----
        qam_tensor (torch.Tensor): A complex-valued tensor of shape
                                   (..., n_symb), containing QAM symbols.
        qam_order (int): QAM modulation order (must be square, e.g., 4, 16).
        device (torch.device): Device to run the computations (CPU or CUDA).
        gray (bool, optional): If True, assumes symbols use Gray encoding and
                               converts back to binary. Default is True.

    Returns
    -------
        torch.Tensor: A tensor of binary bits (0s and 1s) of shape (...,
                      n_symb * n_bits).

    Example:
        >>> symbols = torch.tensor([[-3+3j, 1-1j]], dtype=torch.complex64)
        >>> bits = qam_to_bit(symbols, qam_order=16,
                              device=torch.device("cpu"), gray=True)
        >>> print(bits.shape)  # torch.Size([1, 8])
    """
    # Ensure square QAM, get bits per I/Q component, and power for norm
    n_bits, levels, _ = __square_qam_param(qam_order)

    # Get the I symbols
    data_i = (qam_tensor.real + levels - 1) / 2
    # Get the Q symbols
    data_q = (qam_tensor.imag + levels - 1) / 2

    # Convert from levels to binary
    bit_data_i = __decimal_to_binary(data_i, int(n_bits / 2), device)
    bit_data_q = __decimal_to_binary(data_q, int(n_bits / 2), device)

    # Bit mapping from QAM using Gray mapping
    if gray:

        # Convert from binary to Gray
        bit_data_i = __binary_to_gray(bit_data_i)
        bit_data_q = __binary_to_gray(bit_data_q)

    bit_tensor = cat((bit_data_i, bit_data_q), -1)

    bit_tensor = bit_tensor.flatten(-2, -1)

    return bit_tensor


def __gray_to_binary(bit_tensor):
    """
    Convert a Gray code tensor to binary using vectorized PyTorch operations.

    Args
    ----
        bit_tensor (torch.Tensor): Tensor of 0s and 1s representing Gray code.
                                   Can be of any shape, as long as the last
                                   dimension is the bit sequence. Dtype must be
                                   torch.int8 or torch.long.

    Returns
    -------
        torch.Tensor: Tensor of the same shape as 'bit_tensor', representing
                      the binary code.

    Example
    -------
        >>> gray = torch.tensor([1, 1, 1, 0], dtype=torch.int8)
        >>> binary = gray_to_binary(gray)
        >>> print(binary)  # tensor([1, 0, 1, 1], dtype=torch.int8)
    """
    # Ensure tensor is of integer type
    bit_tensor = bit_tensor.to(int8)

    # Cumulative XOR using binary logic: binary[i] = gray[0] ^ gray[1] ^ ...
    #                                    ^ gray[i]
    binary = cumsum(bit_tensor, dim=-1) % 2

    return binary


def __binary_to_gray(bin_tensor):
    """
    Convert a tensor of binary values to its corresponding Gray code.

    Gray code ensures that only a single bit changes between consecutive
    values, which is useful in digital communications and QAM modulation
    schemes.

    Args
    ----
        bin_tensor (torch.Tensor): A tensor of 0s and 1s (dtype must be int8
                                   or convertible), shape can be (..., n_bits),
                                   where the last dimension is the bit
                                   sequence.

    Returns
    -------
        torch.Tensor: A tensor of the same shape as 'bin_tensor', containing
                      the Gray-coded bits (dtype: int8).

    Example
    -------
        >>> binary = torch.tensor([1, 0, 1, 1], dtype=torch.int8)
        >>> gray = __binary_to_gray(binary)
        >>> print(gray)  # tensor([1, 1, 1, 0], dtype=torch.int8)

        >>> batch = torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.int8)
        >>> __binary_to_gray(batch)
        tensor([[0, 1, 0],
                [1, 1, 0]], dtype=torch.int8)
    """
    # Ensure tensor is of integer type
    bin_tensor = bin_tensor.to(int8)

    # Shifted version of binary (append 0 at the start of each bit sequence)
    shifted = roll(bin_tensor, shifts=1, dims=-1)
    shifted[..., 0] = 0  # Ensure first bit is not wrapped

    # XOR to get Gray code
    gray = bin_tensor ^ shifted

    return gray


def __binary_to_decimal(bit_tensor, device):
    """
    Convert a binary tensor to its decimal (integer) representation.

    This function interprets the last dimension of the input tensor as a binary
    number and converts it into its corresponding decimal value using base-2
    positional weights.

    Args
    ----
        bit_tensor (torch.Tensor): A tensor of shape (..., n_bits) containing
                                   0s and 1s, representing binary numbers along
                                   the last dimension.
        device (torch.device): The device (CPU or GPU) to perform the
                               computation on.

    Returns
    -------
        torch.Tensor: A tensor of decimal integers with shape equal to
                     'bit_tensor.shape[:-1]'.

    Example
    -------
        >>> bits = torch.tensor([[[[1, 0, 1]]]], dtype=torch.int8)
        >>> dec = __binary_to_decimal(bits, device=torch.device("cpu"))
        >>> print(dec)
        tensor([[[5]]])
    """
    powers = 2 ** arange(bit_tensor.size(dim=-1) - 1, -1, -1, device=device)

    decimal = sum_torch(bit_tensor.to(long) * powers, dim=-1)

    return decimal


def __decimal_to_binary(dec_tensor, n_bits, device):
    """
    Convert a tensor of decimal integers to binary representation.

    Args
    ----
        dec_tensor (torch.Tensor): A tensor of non-negative integers.
                                   Shape: (..., 1), can be float or int (will
                                   be cast to int8).
        n_bits (int): Number of bits to represent each decimal number.
        device (torch.device): Device to perform the operation (CPU or CUDA).

    Returns
    -------
        torch.Tensor: A tensor of binary values (0s and 1s) with shape
                      (..., n_bits),
                      dtype: torch.int8. Bits are ordered from MSB to LSB.

    Example
    -------
        >>> dec = torch.tensor([3, 5], dtype=torch.int8)
        >>> __decimal_to_binary(dec, n_bits=4, device=torch.device("cpu"))
        tensor([[0, 0, 1, 1],
                [0, 1, 0, 1]], dtype=torch.int8)
    """
    # Ensure tensor is of integer type
    dec_tensor = dec_tensor.to(int8)

    powers = 2 ** arange(n_bits - 1, -1, -1, device=device)

    # Unsqueeze to match broadcasting: (..., 1)&(num_bits,) => (..., num_bits)
    binary = ((dec_tensor.unsqueeze(-1) & powers) > 0).to(int8)

    return binary


def __square_qam_param(qam_order):
    """
    Return QAM config parameters for a given square QAM modulation order.

    This function provides:
    - The number of bits per symbol ('n_bits')
    - The number of amplitude levels per I/Q axis ('levels')
    - The average constellation power ('power') for normalization

    Args
    ----
        qam_order (int): The modulation order. Must be a supported square QAM
                         value: [2, 4, 16, 64, 256, 1024].

    Returns
    -------
        tuple: A tuple containing:
            - n_bits (int): Total number of bits per QAM symbol.
            - levels (int): Number of amplitude levels per axis (I or Q).
            - power (int): Average constellation power (for normalization).

    Example
    -------
        >>> n_bits, levels, power = __square_qam_param(16)
        >>> print(n_bits, levels, power)
        4 4 10
    """
    # Compute the number of levels
    levels = int(qam_order ** 0.5)

    # Compute the number of bits per QAM symbol
    n_bits = int(log2(tensor(qam_order)))

    axis = arange(levels, dtype=float32)
    axis = 2 * axis - (levels - 1)  # Center constellation at 0

    # Meshgrid over I and Q
    I, Q = meshgrid(axis, axis, indexing='ij')
    constellation = I + 1j * Q

    # Flatten to 1D tensor
    symbols = constellation.reshape(-1)

    # Compute power
    power = float(mean(abs_torch(symbols) ** 2))

    return n_bits, levels, power


def norm_qam_power(qam_tensor, norm='constellation', qam_order=None):
    """
    Normalize a tensor of QAM symbols to unit average power.

    This function supports two modes of normalization:
    - 'constellation': Uses the theoretical average power of a reference square
                       QAM constellation.
    - 'power': Computes and normalizes based on the empirical power of the
               input QAM tensor.

    Args
    ----
        qam_tensor (torch.Tensor): A complex-valued tensor of QAM symbols.
                                   Shape: (..., n_symb).
        norm (str, optional): Normalization method. Options:
                              - 'constellation': Normalize by reference QAM
                                                 constellation power.
                              - 'power': Normalize by empirical average power
                                         of the tensor.
        qam_order (int, optional): QAM order (e.g., 16, 64, 256). Required if
                                   norm='constellation'.

    Returns
    -------
        torch.Tensor: The normalized QAM tensor with approximately unit average
                      power.

    Raises
    ------
        ValueError: If 'norm' mode is invalid or 'qam_order' is missing when
                    required.

    Example:
        >>> qam = torch.tensor([1+1j, -3-3j, 1-1j], dtype=torch.cfloat)
        >>> norm_qam = norm_qam_power(qam, norm='power')
        >>> print(norm_qam)
    """
    # If normalization is performed by reference constellation
    if norm == 'constellation':
        # Ensure square QAM, get bits per I/Q component, and power for norm
        _, _, power = __square_qam_param(qam_order)
        power = tensor(power)

    elif norm == 'power':
        # compute power
        power = mean(abs_torch(qam_tensor)**2, dim=-1, keepdim=True)

    else:
        raise ValueError("Normalization mode not implemented!")

    # Normalization by the constellation power
    qam_tensor = qam_tensor / sqrt(power)

    return qam_tensor


def denorm_qam_power(qam_tensor, qam_order=None):
    """
    Denormalizes a QAM tensor by scaling it to match the theoretical power.

    This function takes a QAM tensor (e.g., previously normalized to unit
    power) and scales it to match the average power of the ideal square QAM
    constellation with the specified 'qam_order'.

    Args
    ----
        qam_tensor (torch.Tensor): A complex-valued tensor of shape
                                   (..., n_symb) containing normalized QAM
                                   symbols.
        qam_order (int): The QAM modulation order (e.g., 16, 64, 256).
                         Required to retrieve the reference power.

    Returns
    -------
        torch.Tensor: A denormalized tensor scaled to match the target
                      constellation power.

    Example:
        >>> qam_norm = torch.tensor([1+1j, -1-1j], dtype=torch.cfloat)
        >>> qam_denorm = denorm_qam_power(qam_norm, qam_order=16)
        >>> print(qam_denorm)
    """
    # Ensure square QAM, get bits per I/Q component, and power for norm
    _, _, power = __square_qam_param(qam_order)
    qam_power = tensor(power)

    # Compute signal power
    sig_power = mean(abs_torch(qam_tensor)**2, dim=-1, keepdim=True)

    # Denormalization by the constellation power
    qam_tensor = qam_tensor * sqrt(qam_power) / sqrt(sig_power)

    return qam_tensor


def quantization_qam(qam_tensor, qam_order, device):
    """
    Quantize a QAM tensor to the nearest ideal constellation points.

    This function assumes square QAM (e.g., 4, 16, 64, 256) and maps each
    received symbol to the closest point in the reference constellation.

    Args
    ----
        qam_tensor (torch.Tensor): Complex tensor of received QAM symbols.
                                   Shape: (..., n_symb).
        qam_order (int): QAM modulation order (must be a perfect square).
        device (torch.device): Device for computation.

    Returns
    -------
        torch.Tensor: Complex tensor of quantized symbols with the same shape
                      as 'qam_tensor'.

    Example:
        >>> qam = torch.tensor([1.2 + 3.1j, -2.8 - 0.9j],dtype=torch.complex64)
        >>> quantized = quantization_qam(qam, qam_order=16,
                                         device=torch.device("cpu"))
        >>> print(quantized)
        tensor([ 1.+3.j, -3.-1.j], dtype=torch.complex64)
    """
    # For square QAM modulations.
    _, n_levels, _ = __square_qam_param(qam_order)
    n_levels = tensor(n_levels)

    # levels of qam symbols (I or Q)
    levels = arange(n_levels, device=device) * 2 - (n_levels - 1)

    # Quantize I symbols
    real_diffs = abs_torch(qam_tensor.real.unsqueeze(-1) - levels)

    quant_i = levels[argmin(real_diffs, dim=-1)]

    # Quantize Q symbols
    imag_diffs = abs_torch(qam_tensor.imag.unsqueeze(-1) - levels)

    quant_q = levels[argmin(imag_diffs, dim=-1)]

    # Join I and Q parts
    quant_symbols = quant_i + 1j*quant_q

    return quant_symbols


def __synchronization(tx_tensor, rx_tensor, device, max_shift=200):
    """
    Compute the optimal time alignment between tx and rx signals.

    Parameters
    ----------
    tx_tensor : torch.Tensor
        Transmitted signal tensor of shape (n_ch, n_pol, n_s), where
        n_ch is the number of channels,
        n_pol is the number of polarizations (usually 2),
        n_s is the number of symbols or samples.

    rx_tensor : torch.Tensor
        Received signal tensor of the same shape as tx_tensor.

    device : torch.device
        The device (CPU or CUDA) on which to perform computations.

    max_shift : int, optional
        The maximum absolute shift (in symbols) to search for synchronization.
        The total search range is [-max_shift, max_shift]. Default is 200.

    Returns
    -------
    pos : torch.Tensor
        Tensor of shape (n_ch, n_pol) containing the optimal shift positions
        that maximize the correlation for each channel and polarization.

    val : torch.Tensor
        Tensor of shape (n_ch,) containing the total correlation values
        (summed over polarizations) for the best shifts in each channel.
    """
    # Get the number of channels, polarizations, and samples
    n_ch, n_pol, n_s = tx_tensor.shape

    # Number os symbols used to compute the correlation
    n_corr = n_s - 2 * max_shift

    # Tensor of correlations
    corr = zeros((n_ch, n_pol, max_shift * 2 + 1), dtype=float32,
                 device=device)

    # Loop through the shifts
    for ii in range(-max_shift, max_shift + 1):
        corr[..., ii + max_shift] = p_corr(tx_tensor[..., max_shift: n_corr],
                                           rx_tensor[..., max_shift + ii:
                                                     n_corr + ii])

    # Positions that maximize
    val, pos = max_torch(corr, dim=-1)

    return pos - max_shift, sum_torch(val, dim=-1)


def synchronization(tx_tensor, rx_tensor, device, max_shift=200):
    """
    Find best synchronization shifts per channel and polarization based on MSE.

    Args
    ----
        tx_tensor (torch.Tensor): Transmitted signal of
                                  shape (n_ch, n_pol, n_s).
        rx_tensor (torch.Tensor): Received signal of same shape.
        device (torch.device): Torch device.
        max_shift (int): Max number of samples to search for alignment.

    Returns
    -------
        shifts (torch.Tensor): Optimal shift per channel and polarization
                               (n_ch, n_pol).
        values (torch.Tensor): Similarity (1 / MSE) per channel (n_ch,).
    """
    # Get the number of channels, polarizations, and samples
    n_ch, n_pol, n_s = tx_tensor.shape

    # Get synchronization without invert polarizations
    pos_d, val_d = __synchronization(tx_tensor, rx_tensor, device, max_shift)

    # Get synchronization inverting polarizations
    pos_i, val_i = __synchronization(tx_tensor, flip(rx_tensor, [-2]), device,
                                     max_shift)

    # Positions to discard to keep both tensors with the same dimension
    n_pos = int(max_torch(tensor([max_torch(abs_torch(pos_d[ch]))
                                  if val_d[ch] >= val_i[ch]
                                  else max_torch(abs_torch(pos_i[ch]))
                                  for ch in range(n_ch)])))

    # Length of output tensors
    n_out = n_s - 2 * n_pos

    # Initialize the output tx and rx
    tx_out = tx_tensor[..., n_pos: n_out + n_pos]
    rx_out = zeros((n_ch, n_pol, n_out), dtype=cfloat, device=device)

    # For each channel
    for ch in range(n_ch):

        if val_d[ch] >= val_i[ch]:

            rx_out[ch, 0, :] = rx_tensor[ch, 0, n_pos + pos_d[ch, 0]:
                                         n_out + n_pos + pos_d[ch, 0]]
            rx_out[ch, 1, :] = rx_tensor[ch, 1, n_pos + pos_d[ch, 1]:
                                         n_out + n_pos + pos_d[ch, 1]]
        else:
            rx_out[ch, 0, :] = rx_tensor[ch, 1, n_pos + pos_i[ch, 0]:
                                         n_out + n_pos + pos_i[ch, 0]]
            rx_out[ch, 1, :] = rx_tensor[ch, 0, n_pos + pos_i[ch, 1]:
                                         n_out + n_pos + pos_i[ch, 1]]

    return tx_out, rx_out


def derotation(tx_tensor, rx_tensor, device):
    """
    Perform phase derotation of the received signal.

    It is performed by testing four quadrature phase shifts (0, π/2, π, 3π/2)
    and selecting the one that minimizes the mean squared error (MSE) with
    respect to the transmitted signal.

    Parameters
    ----------
    tx_tensor : torch.Tensor
        Transmitted complex-valued tensor of shape (n_ch, n_pol, n_samples).

    rx_tensor : torch.Tensor
        Received complex-valued tensor of the same shape as tx_tensor.

    device : torch.device
        Device on which the computation will be performed (CPU or GPU).

    Returns
    -------
    out : torch.Tensor
        Derotated and quantized complex-valued tensor of the same shape as
        input, where the phase rotation that minimizes MSE is applied and
        real/imaginary parts are rounded.
    """
    # tensor of rotations
    rot = tensor([[0], [pi / 2], [pi], [3 * pi / 2]], dtype=cfloat,
                 device=device)

    # Difference of the transmitted data and rotated received data
    dif = tx_tensor.unsqueeze(-2) - (rx_tensor.unsqueeze(-2) * exp(-1j * rot))

    # MSE of rotations
    mse = mean(abs_torch(dif) ** 2, dim=-1)

    # Derotate the received tensor
    out = rx_tensor * exp(-1j * rot[argmin(mse, dim=-1)])

    return round_torch(out.real) + 1j * round_torch(out.imag)
