"""
Created on Tue Apr 15 16:22:19 2025.

@author: Kayol Mayer
"""

from torch.cuda import is_available
from packages.data_streams import random_bit_sequence, bit_to_qam, \
    qam_to_bit, denorm_qam_power, quantization_qam
from packages.sampling import up_sampling, rrc_filter, shaping_filter, \
    matched_filter, down_sampling
from matplotlib.pyplot import scatter, plot

system_par = {
    'n_ch': 10,
    'n_pol': 2,
    'n_bits': int(1e4),
    'rand': 1525,
    'm_qam': 16,
    'gray': True,
    'norm': True,
    'k_up': 16,
    'filt_symb': 20,
    'alpha': 0.2
    }

# Get device to simulate the optical system
# device = "cuda" if is_available() else "cpu"
device = 'cpu'
print(f"Using {device} device")

# Compute the RRC coefficients
filter_coeffs = rrc_filter(system_par['alpha'], system_par['filt_symb'],
                           system_par['k_up'], device)

# Create the bits to be modulated
bit_data_tx = random_bit_sequence(system_par['n_ch'],
                                  system_par['n_pol'],
                                  system_par['n_bits'],
                                  device,
                                  system_par['rand'])

# Create the square QAM symbols from the generated bits
symb_data_tx = bit_to_qam(bit_data_tx, system_par['m_qam'], device,
                          system_par['gray'], system_par['norm'])

# Upsample the symbols to use the shaping filter
symb_data_up = up_sampling(symb_data_tx, system_par['k_up'], device)

# Apply the shaping filter
symb_data_shape = shaping_filter(symb_data_up, filter_coeffs, device)

# Apply the matched filter
symb_data_matched = matched_filter(symb_data_shape, filter_coeffs,
                                   system_par['filt_symb'],
                                   system_par['k_up'], device)

# Downsampling of the symbols
symb_data_down = down_sampling(symb_data_matched, system_par['k_up'])

# Denormalize the QAM signal to the constellation power
symb_data_rx = denorm_qam_power(symb_data_down, qam_order=system_par['m_qam'])

# Quantize the symbols to the reference constellations
symb_data_rx = quantization_qam(symb_data_rx, system_par['m_qam'], device)

# Demodulate symbols to the respective bits
bit_data_rx = qam_to_bit(symb_data_rx, system_par['m_qam'], device, gray=True)

# scatter(symb_data_up[0,0,:].real, symb_data_up[0,0,:].imag)

# plot(symb_data_up_filt_m[0,0,0:1000].real)
