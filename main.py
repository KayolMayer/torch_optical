"""
Created on Tue Apr 15 16:22:19 2025.

@author: Kayol Mayer
"""

from torch.cuda import is_available
from packages.data_streams import random_bit_sequence, bit_to_qam


system_par = {
    'n_ch': 10,
    'n_pol': 2,
    'n_bits': 100000,
    'rand': 1525,
    'm_qam': 1024
    }

# Get device to simulate the optical system
# device = "cuda" if is_available() else "cpu"
device = 'cpu'
print(f"Using {device} device")

# Create the bits to be modulated
bit_data = random_bit_sequence(system_par['n_ch'],
                               system_par['n_pol'],
                               system_par['n_bits'],
                               device,
                               system_par['rand'])


symb_data = bit_to_qam(bit_data, system_par['m_qam'], device)
