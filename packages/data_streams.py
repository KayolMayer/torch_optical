"""
Created on Tue April 15 16:07:28 2025.

@author: Kayol Mayer
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import manual_seed, randint, reshape, arange, uint8, cat, zeros, \
    long, sqrt, tensor, cumsum, roll
from torch import sum as sum_torch
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
        >>> bits = torch.randint(0, 2, (2, 2, 240), dtype=torch.uint8)
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
                                   torch.uint8 or torch.long.

    Returns
    -------
        torch.Tensor: Tensor of the same shape as 'bit_tensor', representing
                      the binary code.

    Example
    -------
        >>> gray = torch.tensor([1, 1, 1, 0], dtype=torch.uint8)
        >>> binary = gray_to_binary(gray)
        >>> print(binary)  # tensor([1, 0, 1, 1], dtype=torch.uint8)
    """
    # Ensure tensor is of integer type
    bit_tensor = bit_tensor.to(uint8)

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
        bin_tensor (torch.Tensor): A tensor of 0s and 1s (dtype must be uint8
                                   or convertible), shape can be (..., n_bits),
                                   where the last dimension is the bit
                                   sequence.

    Returns
    -------
        torch.Tensor: A tensor of the same shape as 'bin_tensor', containing
                      the Gray-coded bits (dtype: uint8).

    Example
    -------
        >>> binary = torch.tensor([1, 0, 1, 1], dtype=torch.uint8)
        >>> gray = __binary_to_gray(binary)
        >>> print(gray)  # tensor([1, 1, 1, 0], dtype=torch.uint8)

        >>> batch = torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.uint8)
        >>> __binary_to_gray(batch)
        tensor([[0, 1, 0],
                [1, 1, 0]], dtype=torch.uint8)
    """
    # Ensure tensor is of integer type
    bin_tensor = bin_tensor.to(uint8)

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
        >>> bits = torch.tensor([[[[1, 0, 1]]]], dtype=torch.uint8)
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
                                   be cast to uint8).
        n_bits (int): Number of bits to represent each decimal number.
        device (torch.device): Device to perform the operation (CPU or CUDA).

    Returns
    -------
        torch.Tensor: A tensor of binary values (0s and 1s) with shape
                      (..., n_bits),
                      dtype: torch.uint8. Bits are ordered from MSB to LSB.

    Example
    -------
        >>> dec = torch.tensor([3, 5], dtype=torch.uint8)
        >>> __decimal_to_binary(dec, n_bits=4, device=torch.device("cpu"))
        tensor([[0, 0, 1, 1],
                [0, 1, 0, 1]], dtype=torch.uint8)
    """
    # Ensure tensor is of integer type
    dec_tensor = dec_tensor.to(uint8)

    powers = 2 ** arange(n_bits - 1, -1, -1, device=device)

    # Unsqueeze to match broadcasting: (..., 1)&(num_bits,) => (..., num_bits)
    binary = ((dec_tensor.unsqueeze(-1) & powers) > 0).to(uint8)

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

    Raises
    ------
        ValueError: If the given modulation order is not supported.

    Example
    -------
        >>> n_bits, levels, power = __square_qam_param(16)
        >>> print(n_bits, levels, power)
        4 4 10
    """
    # Ensure square QAM, get bits per I/Q component, and power for norm
    if qam_order == 2:
        n_bits = 1
        levels = 1
        power = 1
    elif qam_order == 4:
        n_bits = 2
        levels = 2
        power = 2
    elif qam_order == 16:
        n_bits = 4
        levels = 4
        power = 10
    elif qam_order == 64:
        n_bits = 6
        levels = 8
        power = 42
    elif qam_order == 256:
        n_bits = 8
        levels = 16
        power = 170
    elif qam_order == 1024:
        n_bits = 10
        levels = 32
        power = 682
    else:
        raise ValueError("Modulation order not implemented!")

    return n_bits, levels, power
