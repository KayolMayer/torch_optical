"""
Created on Tue Apr 15 16:22:19 2025.

@author: Kayol Mayer
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch.cuda import is_available
from torch import tensor, roll, cat
from packages.data_streams import random_square_qam_sequence, \
    qam_to_bit, denorm_qam_power, quantization_qam, synchronization, \
        derotation, generate_zc_sequences, insert_pilots, synchronization_zc
from packages.sampling import up_sampling, rrc_filter, shaping_filter, \
    matched_filter, down_sampling
from packages.opt_tx import laser_tx, iqModulator
from packages.opt_rx import laser_rx, optical_front_end, insert_skew, adc, \
    deskew, gsop
from packages.amplifier import edfa
from packages.fiber import simple_ssmf
from packages.equalizers import cd_equalization, cma_rde_equalization
from packages.frequency_recovery import freq_rec_4th_power
from packages.phase_recovery import phase_recovery_bps
from packages.metrics import ber_comp, ser_comp
from packages.utils import get_freq_grid, p_corr
from matplotlib.pyplot import figure, scatter, plot, title
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
    'n_zf_seq': 1024,
    'n_blocks': 3,
    'block_len': 1024,
    'pilot_spacing': 32,
    'pmd_eq_convergence_symbs': 50000,  # Symbols considered for convergence
    'rand': 1525,
    'n_spans': 1,
    'm_qam': 16,
    'm_qam_pilot': 4,
    'sr': 30e9,
    'grid_spacing': 50e9,
    'center_freq': 193.1e12,  # Center frequency of the spectrum in Hz
    'gray': True,
    'norm': True,
    'k_up': 16,  # Upsampling factor for RRC
    'filt_symb': 20,
    'alpha': 0.1,  # RRC rolloff
    'tx_laser_power_dbm': 0,
    'tx_laser_lw': 0e3,
    'tx_laser_freq_shift': 0e6,  # Frequency shift in the laser [Hz]
    'rx_laser_power_dbm': 0,
    'rx_laser_lw': 1e6,
    'rx_laser_freq_shift': 0e6,  # Frequency shift in the laser [Hz]
    'vpi': -1,
    'max_exc': -0.8,  # -0.8 * vpi
    'min_exc': -1.2,  # -1.2 * vpi
    'bias': -1,  # -1.0 * vpi
    'responsivity': 1,  # Receiver photodetector responsivity [A/W]
    'phase_shift': 5,  # Phase shift in the hybrid90 [degree]
    'skew': [5e-12, -5e-12, 5e-12, -5e-12],  # Skew for I and Q of each pol [s]
    'adc_f_error_ppm': 0.0,  # frequency error (ppm)
    'adc_phase_error': 0.0,  # phase error [-0.5,0.5]
    'lagrange_order': 10,  # Number of lagrange coefficients (usually 4 to 6)
    'cdc_n_fft': 1024,  # FFT length to compensate CD
    'cdc_fft_overlap': 64,  # Number of samples of overlaping computing the FFT
    'pmd_eq_taps': 15,  # Number of taps of the PMD equalizer
    'pmd_eq_eta': 1e-3,  # Learning rate of the adaptive equalizer
    'pmd_eq_up_samp': 2,  # Equalizer upsampling
    'pmd_eq_batch': 16,  # minibatch during equalization
    'bps_n_symbs': 32,  # Number of symbols to consider in the sum of the BPS
    'bps_n_phases': 64,  # Number of phases to test
    'nf_db_boost': 5.5,  # Booster noise figure in dB
    'gain_db_boost': 22,  # Booster gain in dB
    'fiber_len_km': 50,  # Fiber length [km]
    'fiber_att_db_km': 0.2,  # Fiber attenuation [dB/km]
    'fiber_gamma': 1.27,  # Nonlinear Coefficient [1/W/km]
    'fiber_disp': 17,  # Fiber dispersion [ps/nm/km]
    'fiber_dgd': 0.07,  # Fiber PMD coefficient [ps/√km]
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


# Number of pilots per block
pilots_block = system_par['block_len'] // system_par['pilot_spacing']

# Number of symbols per block
symbols_block = system_par['block_len'] - pilots_block

# Total number of pilots
tot_pilots = pilots_block * system_par['n_blocks']

# Total number of symbols
tot_symbols = symbols_block * system_par['n_blocks']

# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
#
#                               TRANSMITTER SIDE
#
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************

# Create the Zadoff-Chu sequences for preamble
zf_seq = generate_zc_sequences(system_par['n_ch'], system_par['n_pol'],
                               system_par['n_zf_seq'], device)


# Create the 4 QAM pilot symbols
symb_data_pilots = random_square_qam_sequence(system_par['n_ch'],
                                              system_par['n_pol'],
                                              tot_pilots,
                                              system_par['m_qam_pilot'],
                                              device,
                                              system_par['rand'],
                                              system_par['gray'],
                                              system_par['norm'])

# Create the 16 QAM symbols for payload
symb_data_tx = random_square_qam_sequence(system_par['n_ch'],
                                          system_par['n_pol'],
                                          tot_symbols,
                                          system_par['m_qam'],
                                          device,
                                          system_par['rand'],
                                          system_par['gray'],
                                          system_par['norm'])

# Insert pilots in the sequence
symbs_tx = insert_pilots(symb_data_tx, symb_data_pilots,
                         system_par['pilot_spacing'], device)

# Insert preamble
symb_complete_tx = cat((zf_seq, symbs_tx), -1)

# Upsample the symbols to use the shaping filter
symb_data_up = up_sampling(symb_complete_tx, system_par['k_up'], device)

# Apply the shaping filter
symb_data_shape = shaping_filter(symb_data_up, filter_coeffs, device)

# Create the laser source. All channels are considered centered in zero, but
# the impairments are applied to their respective frequency.
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

# Preamp
sig_ch = edfa(sig_ch, system_par['nf_db_boost'], system_par['gain_db_boost'],
              system_par['center_freq'] + freq_grid, system_par['sr'],
              system_par['alpha'], system_par['k_up'], system_par['rand'],
              device)

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

# Exclude transient samples
signal_skew = signal_skew[..., int((system_par['filt_symb'] *
                                    system_par['k_up']) / 2):]

# Downsampling to 2 samples per symbol ADC
symb_data_ds = down_sampling(signal_skew,
                             int(system_par['k_up'] /
                                 system_par['pmd_eq_up_samp']))

# Deskew with Lagrange interpolator
symb_data_deskew = deskew(symb_data_ds, system_par['pmd_eq_up_samp'],
                          system_par['sr'], system_par['lagrange_order'],
                          system_par['skew'], device)

# Gram-Schmidt Orthogonalization
symb_data_gsop = gsop(symb_data_deskew)

# Syncronization
symb_complete_rx = synchronization_zc(up_sampling(zf_seq,
                                                  system_par['pmd_eq_up_samp'],
                                                  device),
                                      symb_data_gsop, device)[..., :
                                                              2 *
                                                              symb_complete_tx.shape[-1]]
