"""
Created on Tue April 15 16:07:28 2025.

@author: Kayol Mayer
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import manual_seed, randint, reshape, arange, uint8, cat, zeros, \
    long, sqrt, tensor
from torch import sum as sum_torch
# =============================================================================
# =============================================================================

# *****************************************************************************
# *****************************************************************************


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
    Convert a binary bit sequence into a complex QAM symbol sequence.

    Supports both Gray-coded and non-Gray-coded mapping for square QAM
    constellations (e.g., 4-QAM, 16-QAM, etc.).
    Optionally normalizes the QAM constellation to unit average power.

    Args
    ----
        bit_sequence (numpy.ndarray): Input bit sequence as a 1D array of uint8 values (e.g., [0, 1, 1, 0, ...]).
        order (int): QAM order (must be a perfect square, e.g., 4, 16, 64).
        gray (bool, optional): Whether to use Gray coding for symbol mapping. Default is True.
        norm (bool, optional): Whether to normalize output symbols to unit average power. Default is True.

    Returns
    -------
        numpy.ndarray: Complex-valued QAM symbol sequence ('dtype=complex128').

    Example:
        >>> import numpy as np
        >>> bits = np.random.randint(0, 2, size=64, dtype=np.uint8)
        >>> symbols = bit_sequence_to_qam(bits, order=16, gray=True)
        >>> symbols[:5]
        array([-0.9486833 +0.9486833j ,  0.31622777-0.9486833j , ...])
    """
    # Ensure square QAM and compute bits per I/Q component
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

    # Number of bit blocks to fit the number of symbols
    n_blocks = int(bit_tensor.size(dim=2)//n_bits)

    # Bit tensor after clipping extra bits
    bit_tensor = bit_tensor[:, :, 0:n_blocks * n_bits]

    # Split the bit sequence into blocks for I/Q symbol generation
    bit_blocks = reshape(bit_tensor, (bit_tensor.size(dim=0),
                                      bit_tensor.size(dim=1),
                                      n_blocks, n_bits))

    if gray:
        # Convert bit blocks to Gray mapping
        bit_blocks = __binary_to_gray(bit_blocks, device)

    # Convert from binary to decimal
    decimal_i = __binary_to_decimal(bit_blocks[:, :, :, :int(n_bits/2)],
                                    device)
    decimal_q = __binary_to_decimal(bit_blocks[:, :, :, int(n_bits/2):],
                                    device)

    # Convert from decimal to QAM
    qam = (decimal_i * 2 - (levels - 1)) + 1j * (decimal_q * 2 - (levels - 1))

    if norm:

        qam = qam / sqrt(tensor(power))

    return qam


def __binary_to_gray(bit_tensor, device):
    """
    Convert a binary bit sequence (0 and 1) into its Gray code representation.

    Args
    ----
        bit_tensor (torch.Tensor): A 1D tensor of binary bits
                                  (dtype=torch.uint8 or torch.long).

    Returns
    -------
        torch.Tensor: A 1D tensor representing the Gray code of the input bits.

    Example
    -------
        >>> bits = torch.tensor([1, 0, 1, 1], dtype=torch.uint8)
        >>> gray = binary_to_gray(bits)
        >>> print(gray)  # Output: tensor([1, 1, 1, 0])
    """
    # Ensure tensor is of integer type
    bit_tensor = bit_tensor.to(uint8)

    # Shift bits to the right by 1 (prepend 0)
    shifted = cat([zeros(bit_tensor.size(dim=0), bit_tensor.size(dim=1),
                         bit_tensor.size(dim=2), 1,
                         dtype=uint8, device=device),
                   bit_tensor[:, :, :, :-1]], -1)

    # XOR with original
    gray_code = bit_tensor ^ shifted

    return gray_code


def __binary_to_decimal(bit_tensor, device):
    """
    Converts a 1D binary tensor (sequence of 0s and 1s) to its decimal integer value.

    Args:
        bit_tensor (torch.Tensor): A 1D tensor of bits (dtype: torch.uint8 or torch.long),
                                   e.g., tensor([1, 0, 1, 1]) for binary '1011'.

    Returns:
        int: The decimal equivalent of the binary number.

    Example:
        >>> bits = torch.tensor([1, 0, 1, 1], dtype=torch.uint8)
        >>> decimal = binary_tensor_to_decimal(bits)
        >>> print(decimal)  # Output: 11
    """
    powers = arange(bit_tensor.size(dim=-1) - 1, -1, -1, device=device)

    decimal = sum_torch(bit_tensor.to(long)*(2**powers), dim=-1)

    return decimal
