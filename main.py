"""
Created on Tue Apr 15 16:22:19 2025.

@author: Kayol Mayer
"""

from torch.cuda import is_available
from torch import tensor
from packages.data_streams import random_bit_sequence, bit_to_qam, \
    qam_to_bit, denorm_qam_power, quantization_qam
from packages.sampling import up_sampling, rrc_filter, shaping_filter, \
    matched_filter, down_sampling
from packages.opt_tx import laser_tx, iqModulator, mux
from packages.opt_rx import laser_rx, optical_front_end, insert_skew, adc, \
    deskew, gsop
from packages.amplifier import edfa
from packages.fiber import ssmf, simple_ssmf
from packages.equalizers import cd_equalization
from packages.utils import get_freq_grid
from matplotlib.pyplot import scatter, plot


# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
#
#                     SIMULATION PARAMETERS AND VARIABLES
#
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************

system_par = {
    'n_ch': 3,
    'n_pol': 2,
    'n_bits': 10000,
    'rand': 1525,
    'm_qam': 16,
    'sr': 40e9,
    'grid_spacing': 50e9,
    'center_freq': 193.1e12,  # Center frequency of the spectrum in Hz
    'gray': True,
    'norm': True,
    'k_up': 16,  # Upsampling factor for RRC
    'filt_symb': 20,
    'alpha': 0.2,  # RRC rolloff
    'tx_laser_power_dbm': 0,
    'tx_laser_lw': 20e3,
    'rx_laser_power_dbm': 0,
    'rx_laser_lw': 20e3,
    'vpi': -1,
    'max_exc': -0.8,  # -0.8 * vpi
    'min_exc': -1.2,  # -1.2 * vpi
    'bias': -1,  # -1.0 * vpi
    'responsivity': 1,  # Receiver photodetector responsivity [A/W]
    'phase_shift': 5,  # Phase shift in the hybrid90 [degree]
    'skew': [5e-12, -5e-12, 5e-12, -5e-12],  # Skew for I and Q of each pol [s]
    'adc_samples': 2,  # Number of samples after the ADC
    'adc_f_error_ppm': 0.0,  # frequency error (ppm)
    'adc_phase_error': 0.0,  # phase error [-0.5,0.5]
    'lagrange_order': 10,  # Number of lagrange coeeficients (usually 4 to 6)
    'cdc_n_fft': 1024,  # FFT length to compensate CD
    'cdc_fft_overlap': 64,  # Number of samples of overlaping computing the FFT
    'nf_db_boost': 5.5,  # Booster noise figure in dB
    'gain_db_boost': 20,  # Booster gain in dB
    'fiber_len_km': 80,  # Fiber length [km]
    'fiber_att_db_km': 0.2,  # Fiber attenuation [dB/km]
    'fiber_gamma': 1.27,  # Nonlinear Coefficient [1/W/km]
    'fiber_disp': 17,  # Fiber dispersion [ps/nm/km]
    'fiber_dgd': 0.1,  # Fiber PMD coefficient [ps/√km]
    }

# Get device to simulate the optical system
# device = "cuda" if is_available() else "cpu"
device = 'cpu'
print(f"Using {device} device")

# Compute the RRC coefficients
filter_coeffs = rrc_filter(system_par['alpha'], system_par['filt_symb'],
                           system_par['k_up'], device)

# Get the frequency grid centered at 0 Hz
freq_grid = get_freq_grid(system_par['n_ch'], system_par['grid_spacing'])

# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
#
#                               TRANSMITTER SIDE
#
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************

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

# Create the laser source. All channels are considered centered in zero, but
# the impairments are applied to their rescpective frequency.
laser_tx = laser_tx(system_par['n_ch'], system_par['n_pol'],
                    symb_data_shape.shape[-1],
                    system_par['tx_laser_power_dbm'],
                    system_par['sr'], system_par['k_up'],
                    system_par['tx_laser_lw'], tensor(0),
                    system_par['rand'], device)

# Apply the signal of all channels to the IQ modulator
sig_tx = iqModulator(symb_data_shape, laser_tx, system_par['max_exc'],
                     system_par['min_exc'], system_par['bias'],
                     system_par['vpi'])

# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
#
#                            TRANSMISSION CHANNEL
#
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************

# Booster
sig_ch = edfa(sig_tx, system_par['nf_db_boost'], system_par['gain_db_boost'],
              system_par['center_freq'] + freq_grid, system_par['sr'],
              system_par['alpha'], system_par['k_up'], system_par['rand'],
              device)

# Fiber
#sig_ch = ssmf(sig_ch, system_par['center_freq'] + freq_grid, device,
#              **system_par)
sig_ch = simple_ssmf(sig_ch, system_par['center_freq'] + freq_grid, device,
                     **system_par)

# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
#
#                               RECEIVER SIDE
#
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************

# Create the laser source. All channels are considered centered in zero, but
# the impairments are applied to their rescpective frequency.
laser_rx = laser_rx(system_par['n_ch'], system_par['n_pol'],
                    symb_data_shape.shape[-1],
                    system_par['rx_laser_power_dbm'],
                    system_par['sr'], system_par['k_up'],
                    system_par['rx_laser_lw'], tensor(0),
                    system_par['rand']*1000, device)

# Apply the optical front end to recover the vertical and horizontal pols
signal_rx = optical_front_end(sig_ch, laser_rx, system_par['responsivity'],
                              system_par['phase_shift'], device)

# Apply skew as a source of problem
signal_skew = insert_skew(signal_rx, system_par['k_up'], system_par['sr'],
                          system_par['skew'], device)

# Apply the matched filter
symb_data_matched = matched_filter(signal_skew, filter_coeffs,
                                   system_par['filt_symb'],
                                   system_par['k_up'], device)

symb_data_adc = adc(symb_data_matched, system_par['k_up'],
                    system_par['adc_samples'], system_par['adc_f_error_ppm'],
                    system_par['adc_phase_error'], device)

# Deskew with Lagrange interpolator
symb_data_deskew = deskew(symb_data_adc, system_par['adc_samples'],
                          system_par['sr'], system_par['lagrange_order'],
                          system_par['skew'], device)

# Gram-Schmidt Orthogonalization
symb_data_gsop = gsop(symb_data_deskew)

# CD compensation
symb_data_cdc = cd_equalization(symb_data_gsop, system_par['fiber_disp'],
                                system_par['fiber_len_km'],
                                system_par['center_freq'] + freq_grid,
                                system_par['sr'],
                                system_par['adc_samples'],
                                system_par['cdc_n_fft'],
                                system_par['cdc_fft_overlap'], device)

# Downsampling of the symbols
symb_data_down = down_sampling(symb_data_cdc, system_par['adc_samples'])

# Denormalize the QAM signal to the constellation power
symb_data_rx = denorm_qam_power(symb_data_down, qam_order=system_par['m_qam'])

# Quantize the symbols to the reference constellations
#symb_data_rx = quantization_qam(symb_data_rx, system_par['m_qam'], device)

# Demodulate symbols to the respective bits
#bit_data_rx = qam_to_bit(symb_data_rx, system_par['m_qam'], device, gray=True)

scatter(symb_data_rx.real, symb_data_rx.imag)

# plot(symb_data_up_filt_m[0,0,0:1000].real)
