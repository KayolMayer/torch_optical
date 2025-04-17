"""
Created on Tue Apr 15 16:22:19 2025.

@author: Kayol Mayer
"""

from torch.cuda import is_available
from packages.data_streams import random_bit_sequence, bit_to_qam, qam_to_bit,\
    norm_qam_power, denorm_qam_power, quantization_qam
from matplotlib.pyplot import scatter

system_par = {
    'n_ch': 10,
    'n_pol': 2,
    'n_bits': int(1e5),
    'rand': 1525,
    'm_qam': 16
    }

# Get device to simulate the optical system
# device = "cuda" if is_available() else "cpu"
device = 'cpu'
print(f"Using {device} device")

# Create the bits to be modulated
bit_data_tx = random_bit_sequence(system_par['n_ch'],
                                  system_par['n_pol'],
                                  system_par['n_bits'],
                                  device,
                                  system_par['rand'])


symb_data = bit_to_qam(bit_data_tx, system_par['m_qam'], device,gray=True,norm=True)

#symb_data = norm_qam_power(symb_data, norm='power',  qam_order=system_par['m_qam'])

symb_data = denorm_qam_power(symb_data, qam_order=system_par['m_qam'])

symb_data = quantization_qam(symb_data, system_par['m_qam'], device)

#bit_data_rx = qam_to_bit(symb_data, system_par['m_qam'], device, gray=True)

scatter(symb_data[0,0,:].real, symb_data[0,0,:].imag)
