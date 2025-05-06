"""
Created on Tue Apr 15 16:22:19 2025.

@author: Kayol Mayer
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch.cuda import is_available
from torch import tensor
from packages.data_streams import random_square_qam_sequence, \
    qam_to_bit, denorm_qam_power, quantization_qam, synchronization, derotation
from packages.sampling import up_sampling, rrc_filter, shaping_filter, \
    matched_filter
from packages.opt_tx import laser_tx, iqModulator
from packages.opt_rx import laser_rx, optical_front_end, insert_skew, adc, \
    deskew, gsop
from packages.amplifier import edfa
from packages.fiber import simple_ssmf
from packages.equalizers import cd_equalization, cma_rde_equalization
from packages.frequency_recovery import freq_rec_4th_power
from packages.phase_recovery import phase_recovery_bps
from packages.metrics import ber_comp, ser_comp
from packages.utils import get_freq_grid
# =============================================================================
# =============================================================================

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
    'n_ch': 1,
    'n_pol': 2,
    'n_symbols': 100000,
    'pmd_eq_convergence_symbs': 50000,  # Symbols considered for convergence
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
    'tx_laser_lw': 30e3,
    'tx_laser_freq_shift': 0e6,  # Frequency shift in the laser [Hz]
    'rx_laser_power_dbm': 0,
    'rx_laser_lw': 30e3,
    'rx_laser_freq_shift': 100e6,  # Frequency shift in the laser [Hz]
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
    'lagrange_order': 10,  # Number of lagrange coefficients (usually 4 to 6)
    'cdc_n_fft': 1024,  # FFT length to compensate CD
    'cdc_fft_overlap': 64,  # Number of samples of overlaping computing the FFT
    'pmd_eq_taps': 15,  # Number of taps of the PMD equalizer
    'pmd_eq_eta': 1e-4,  # Learning rate of the adaptive equalizer
    'bps_n_symbs': 32,  # Number of symbols to consider in the sum of the BPS
    'bps_n_phases': 64,  # Number of phases to test
    'nf_db_boost': 5.5,  # Booster noise figure in dB
    'gain_db_boost': 22,  # Booster gain in dB
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

# Create the square QAM symbols
symb_data_tx = random_square_qam_sequence(system_par['n_ch'],
                                          system_par['n_pol'],
                                          system_par['n_symbols'],
                                          system_par['m_qam'],
                                          device,
                                          system_par['rand'],
                                          system_par['gray'],
                                          system_par['norm'])

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
                    system_par['tx_laser_freq_shift'],
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
                    system_par['rx_laser_freq_shift'],
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

# CD compensation (Static Equalization)
symb_data_cdc = cd_equalization(symb_data_gsop, system_par['fiber_disp'],
                                system_par['fiber_len_km'],
                                system_par['center_freq'] + freq_grid,
                                system_par['sr'],
                                system_par['adc_samples'],
                                system_par['cdc_n_fft'],
                                system_par['cdc_fft_overlap'], device)

# PMD Equalization (Dynamic Equalization)
symb_data_pmdc = cma_rde_equalization(symb_data_cdc, system_par['adc_samples'],
                                      system_par['pmd_eq_taps'],
                                      system_par['pmd_eq_eta'],
                                      system_par['pmd_eq_convergence_symbs'],
                                      system_par['m_qam'],
                                      system_par['norm'], device)

# Frequency recovery with 4-th power algorithm
symb_data_fr = freq_rec_4th_power(symb_data_pmdc,
                                  system_par['sr'],
                                  system_par['pmd_eq_convergence_symbs'],
                                  device)

# Phase recovery
symb_data_pr = phase_recovery_bps(symb_data_fr, system_par['m_qam'],
                                  system_par['bps_n_symbs'],
                                  system_par['bps_n_phases'], device)

# Denormalize the QAM signal to the constellation power
symb_data_rx = denorm_qam_power(symb_data_pr, qam_order=system_par['m_qam'])

# Quantize the symbols to the reference constellations
symb_data_rx = quantization_qam(symb_data_rx, system_par['m_qam'], device)

# Denormalize the QAM signal to the constellation power
symb_data_tx = denorm_qam_power(symb_data_tx, qam_order=system_par['m_qam'])

# Quantize the symbols to the reference constellations
symb_data_tx = quantization_qam(symb_data_tx, system_par['m_qam'], device)

# Synchronized sequences
tx_sync, rx_sync = synchronization(symb_data_tx, symb_data_rx, device)

# Discard symbols before convergence (equalizer) and in the end (BPS)
tx_sync = tx_sync[..., system_par['pmd_eq_convergence_symbs']:
                  -int(system_par['bps_n_phases'] / 2)]
rx_sync = rx_sync[..., system_par['pmd_eq_convergence_symbs']:
                  -int(system_par['bps_n_phases'] / 2)]

# Derotate Rx streams
rx_sync = derotation(tx_sync, rx_sync, device)

# Demodulate symbols to the respective bits
bit_tx_sync = qam_to_bit(tx_sync, system_par['m_qam'], device, gray=True)
bit_rx_sync = qam_to_bit(rx_sync, system_par['m_qam'], device, gray=True)

# SER computation
ser = ser_comp(tx_sync, rx_sync)

# BER computation
ber = ber_comp(bit_tx_sync, bit_rx_sync)
