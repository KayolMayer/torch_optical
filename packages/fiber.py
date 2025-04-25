"""
Created on Thu Apr 24 13:35:51 2025.

@author: kayol
"""

# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from torch import sqrt, exp, log, tensor, float32, pi, clone, floor, randn, \
    cfloat, arange
from torch import sum as sum_torch
from torch import abs as abs_torch
from torch import max as max_torch
from torch import min as min_torch
from torch.fft import fftfreq, ifft, fft, ifftshift, fftshift
from torch.linalg import vector_norm, svd
# =============================================================================
# =============================================================================


def ssmf(signal, fc, device, **kwargs):
    """
    Simulate signal propagation through a Standard Single Mode Fiber (SSMF).

    It automatically selects the appropriate SSFM solver based on the number
    of polarizations (1 or 2).

    Args
    ----
        signal (torch.Tensor): Complex input signal of
                               shape (n_ch, n_pol, n_samples)
        fc (torch.Tensor): Tensor of frequencies (n_ch)
        device (torch.device): Device for computation
        **kwargs:
            sr (float): Symbol rate [Hz]
            k_up (int): Upsampling factor (samples per symbol)
            fiber_len_km (float): Fiber length [km]
            fiber_att_db_km (float): Attenuation [dB/km]
            fiber_disp (float): Dispersion parameter β2 [s²/m]
            fiber_gamma (float): Nonlinear coefficient [1/W/km]

    Returns
    -------
        torch.Tensor: Output signal of the same shape as input

    Raises
    ------
        ValueError: If the number of polarizations is not 1 or 2
    """
    # Get the number of polarizations
    n_pol = signal.shape[1]

    # If the number of polarizations is one, use the SSFM
    if n_pol == 1:
        out = ssfm(signal, kwargs['sr'], kwargs['k_up'],
                   kwargs['fiber_len_km'], kwargs['fiber_att_db_km'],
                   kwargs['fiber_disp'], kwargs['fiber_gamma'], fc, device)
    elif n_pol == 2:
        out = manakovSSF(signal, kwargs['sr'], kwargs['k_up'],
                         kwargs['fiber_len_km'], kwargs['fiber_att_db_km'],
                         kwargs['fiber_disp'], kwargs['fiber_dgd'],
                         kwargs['fiber_gamma'], fc, device)
    else:
        ValueError('The input signal must have only one or two polarizations!')

    return out


def simple_ssmf(signal, fc, device, **kwargs):
    """
    Simulate signal propagation through a simplified SSMF using linear effects.

        - Fiber attenuation (linear, power-based)
        - Chromatic dispersion (CD)
        - Polarization Mode Dispersion (PMD, only if dual-pol)

    This version does NOT include nonlinear effects.

    Args
    ----
        signal (torch.Tensor): Input signal of shape (n_ch, n_pol, n_samples)
        fc (float): Carrier frequency [Hz]
        device (torch.device): PyTorch device (e.g. "cuda")
        **kwargs:
            fiber_len_km (float): Fiber length [km]
            fiber_att_db_km (float): Fiber attenuation [dB/km]
            fiber_disp (float): Dispersion parameter D [ps/nm/km]
            fiber_dgd (float): PMD coefficient [ps/√km]
                               (used only if n_pol == 2)
            sr (float): Symbol rate [Hz]
            k_up (int): Upsampling factor (samples per symbol)

    Returns
    -------
        torch.Tensor: Output signal after fiber effects (same shape as input)
    """
    # Get the number of polarizations
    n_pol = signal.shape[1]

    # Apply nonlinearity
    signal = __nonlinear_phase_shift(signal, kwargs['fiber_len_km'],
                                     kwargs['fiber_gamma'], device)

    # Apply attenuation
    signal = __att_fiber(signal, kwargs['fiber_len_km'],
                         kwargs['fiber_att_db_km'], device)

    # Transform signal to the frequency domain
    signal = fft(signal)

    # Apply CD
    signal = __cd(signal, kwargs['fiber_disp'], kwargs['fiber_len_km'], fc,
                  kwargs['sr'], kwargs['k_up'], device)

    # If the number of polarizations is two, apply PMD
    if n_pol == 2:
        signal = __pmd(signal, kwargs['fiber_dgd'], kwargs['fiber_len_km'],
                       int(2 * kwargs['fiber_len_km']), kwargs['sr'],
                       kwargs['k_up'], device)

    # Transform signal to the time domain
    signal = ifft(signal)

    return signal


def ssfm(signal, sr, k, L, att_db, D, gamma, fc, device, hz=0.5):
    """
    Simulate optical signal propagation through a single-polarization.

    It considers a standard single-mode fiber using the Split-Step Fourier
    Method (SSFM).

    The model includes:
    - Chromatic dispersion
    - Kerr nonlinearity (self-phase modulation)
    - Fiber attenuation (linear, in Nepper/km)

    Args
    ----
        signal (torch.Tensor): Input signal of shape (n_ch, 1, n_samples)
        sr (float): Symbol rate [Hz]
        k (int): Upsampling factor (samples per symbol)
        L (float): Total fiber length [km]
        att_db (float): Attenuation coefficient [dB/km]
        D (float): Dispersion coefficient [ps/nm/km]
        gamma (float): Nonlinear coefficient [1/W/km]
        fc (torch.Tensor): Tensor of frequencies (n_ch)
        device (torch.device): PyTorch device (e.g., "cuda")
        hz (float, optional): Step size in km for the SSFM loop
                             (default is 0.5)

    Returns
    -------
        torch.Tensor: Output signal with same shape as input
                      (n_ch, 1, n_samples)

    Notes
    -----
        - Implements symmetric SSFM: linear -> nonlinear -> linear per step
        - The dispersion parameter β₂ is derived from D and fc
        - Attenuation is applied per step using exponential decay in amplitude
    """
    # Speed of light in km/s
    c = 299792.458

    # wavelength carrier frequency [km]
    lambdaC = c / fc

    # Fiber attenuation [Nepper/km]
    att_lin = att_db * log(tensor(10, dtype=float32, device=device)) / 10

    # D [ps/nm/km]
    beta2 = -((D * lambdaC ** 2) / (2 * pi * c)).view(-1, 1, 1)

    # Sampling rate [Hz]
    fs = sr * k

    # Number of channels, polarizations, and samples
    n_ch, n_pol, n_s = signal.shape

    # Frequency axis
    omega = 2 * pi * fs * fftfreq(n_s, device=device)

    # single-polarization field
    E = fft(signal)

    # define the static part of the linear operator
    lin_operator = exp(-(att_lin*hz / 2) + 1j * (beta2 * hz / 2) * omega ** 2)

    # Number of steps during iterations
    Nsteps = int(floor(tensor(L / hz)))

    # fiber propagation step
    for _ in range(Nsteps):

        # First linear step (frequency domain)
        E = E * lin_operator

        # Nonlinear step (time domain)
        E = ifft(E)
        E = E * exp(1j * gamma * E * E.conj() * hz)

        # Second linear step (frequency domain)
        E = fft(E)
        E = E * lin_operator

    # Get back to time domain
    E = ifft(E)

    return E


def __apply_pmd(E, omega, tau, device):
    """
    Apply a polarization mode dispersion (PMD) to a dual-polarization signal.

    It is performed by rotating into a random birefringence basis, applying
    frequency-dependent phase delay, and rotating back.

    Args
    ----
        E (torch.Tensor): Optical field (n_ch, 2, n_freq)
        omega (torch.Tensor): Angular frequency vector (n_freq,)
        tau (float): Differential Group Delay [seconds]
        device (torch.device): Target device for computation

    Returns
    -------
        E (torch.Tensor): Signal after DGD rotation
    """
    # Get the number of channels
    n_ch = E.shape[0]

    # Create the unitary matrices for rotations
    U, _, Vh = svd(randn(n_ch, 2, 2, dtype=cfloat, device=device))

    # Apply rotations
    Erot = Vh @ E

    # Applying DGD
    phase = exp(tensor([[1j], [-1j]], dtype=cfloat, device=device) *
                omega * tau / 2)

    Erot = phase * Erot

    # Apply rotations
    E = U @ Erot

    return E


def manakovSSF(signal, sr, k, L, att_db, D, dgd, gamma, fc, device, maxIter=10,
               hz=0.5, tol=1e-5, nlprMethod=True, maxNlinPhaseRot=2e-2):
    """
    Simulate dual-polarization optical signal propagation through a fiber.

    It uses the Manakov equation and the Split-Step Fourier Method (SSFM).

    The method models:
        - Chromatic dispersion
        - PMD
        - Fiber attenuation
        - Kerr nonlinearity (SPM + XPM via Manakov approximation)
        - Adaptive step size based on nonlinear phase rotation (optional)

    Args
    ----
        signal (torch.Tensor): Input signal of shape (n_ch, 2, n_samples)
        sr (float): Symbol rate [Hz]
        k (int): Upsampling factor (samples per symbol)
        L (float): Fiber length [km]
        att_db (float): Fiber attenuation [dB/km]
        D (float): Chromatic dispersion [ps/nm/km]
        dgd (float): Dispersive group delay [ps/√km]
        gamma (float): Nonlinear coefficient [1/W/km]
        fc (float): Carrier frequency [Hz]
        device (torch.device): Torch device for computation
        maxIter (int, optional): Max iterations for trapezoidal nonlinear
                                 solver (default: 10)
        hz (float, optional): Step size [km] if nlprMethod is False
                              (default: 0.5)
        tol (float, optional): Tolerance for nonlinear convergence
                               (default: 1e-5)
        nlprMethod (bool, optional): Enable adaptive step sizing based on
                                     nonlinear phase rotation (default: True)
        maxNlinPhaseRot (float, optional): Max allowed nonlinear phase rotation
                                           per step [rad] (default: 2e-2)

    Returns
    -------
        torch.Tensor: Output signal after propagation,
                      shape (n_ch, 2, n_samples)

    Raises
    ------
        ValueError: If maximum allowed tolerance is not achieved during
                    nonlinear iterations.

    Notes
    -----
        - Implements trapezoidal integration for improved nonlinear accuracy.
        - Adaptive step size improves efficiency and accuracy in highly
          nonlinear regimes.
        - Assumes PMD is negligible or averaged out by the Manakov
          approximation.
    """
    # Speed of light in km/s
    c = 299792.458

    # Convert the length of the fiber to a tensor
    L = tensor(L, dtype=float32, device=device)

    # wavelength carrier frequency [km]
    lambdaC = c / fc

    # Fiber attenuation [Nepper/km]
    att_lin = att_db * log(tensor(10, dtype=float32, device=device)) / 10

    # D [ps/nm/km]
    beta2 = -((D * lambdaC ** 2) / (2 * pi * c)).view(-1, 1, 1)

    # Sampling rate [Hz]
    fs = sr * k

    # Standard deviation of the Maxwellian distribution
    dgd_std = sqrt(tensor(3 * pi / 8, dtype=float32, device=device)) * dgd

    # Number of channels, polarizations, and samples
    n_ch, n_pol, n_s = signal.shape

    # Frequency axis
    omega = 2 * pi * fs * fftfreq(n_s, device=device)

    # define the static part of the linear operator
    argLinOp = - att_lin / 2 + 1j * beta2 / 2 * omega ** 2

    # Copy the vertical and horizontal fields
    E = clone(signal)
    E_conv = clone(signal)

    # Define the z position in the fiber
    z_current = 0

    # Iterations throughout the fiber
    while z_current < L:

        # Power
        power = sum_torch(E * E.conj(), dim=1, keepdim=True)
        power_c = sum_torch(E_conv * E_conv.conj(), dim=1, keepdim=True)

        # Calculate nonlinear phase-shift per step for the Manakov SSFM
        phiRot = ((8 / 9) * gamma * (power + power_c) / 2).real

        # Adaptive step size
        if nlprMethod:
            hz_ = min_torch(maxNlinPhaseRot / max_torch(phiRot), L - z_current)
        # check that the remaining  distance is not less than hz (due to
        # non-integer steps/span)
        elif L - z_current < hz:
            hz_ = L - z_current
        else:
            hz_ = hz

        # define the linear operator
        linOperator = exp(argLinOp * (hz_ / 2))

        # First linear step (frequency domain)
        E_hd = ifft(fft(E) * linOperator)

        # Nonlinear step (time domain)
        for nIter in range(maxIter):

            rotOperator = exp(1j * phiRot * hz_)

            # Second linear step (frequency domain)
            E_fd = ifft(fft(E_hd * rotOperator) * linOperator)

            # check convergence o trapezoidal integration in phiRot
            lim = sqrt(vector_norm(E_fd-E_conv) ** 2) / \
                sqrt(vector_norm(E_conv) ** 2)

            E_conv = clone(E_fd)

            if lim < tol:
                break
            elif nIter == maxIter - 1:
                ValueError('Target SSFM error tolerance was not achieved')

            # Calculate nonlinear phase-shift per step for the Manakov SSFM
            power_c = sum_torch(E_conv * E_conv.conj(), dim=1, keepdim=True)

            # Calculate nonlinear phase-shift per step for the Manakov SSFM
            phiRot = ((8 / 9) * gamma * (power + power_c) / 2).real

        E = clone(E_fd)

        # update propagated distance
        z_current += hz_

        # PMD of the Standard deviation of the Maxwellian distribution per
        # section
        tau = dgd_std * sqrt(hz_) * 1e-12  # s/√km to s

        # PMD in the frequency domain
        E = ifft(__apply_pmd(fft(E), omega, tau, device))

    return E


def __pmd(E_freq, dgd, fiber_len, n_sec, sr, k, device):
    """
    Apply PMD via multiple birefringent sections in the frequency domain.

    Args
    ----
        E_freq (torch.Tensor): Frequency-domain dual-pol signal (n_ch, 2, n_s)
        dgd (float): PMD coefficient [ps/√km]
        fiber_len (float): Fiber length [km]
        n_sec (int): Number of random PMD sections
        sr (float): Symbol rate [Hz]
        k (int): Oversampling factor
        device (torch.device): Target device

    Returns
    -------
        torch.Tensor: PMD-affected frequency-domain signal (n_ch, 2, n_s)
    """
    # Get the number of channels and samples
    n_ch, _, n_s = E_freq.shape

    # Create the unitary matrices for rotations
    U, _, Vh = svd(randn(n_sec, n_ch, 2, 2, dtype=cfloat, device=device))

    # Get the frequency vector
    omega = 2 * pi * fftfreq(n_s, device=device) * k * sr

    # Standard deviation of the Maxwellian distribution
    dgd_std = sqrt(tensor(3 * pi / 8, dtype=float32, device=device)) * dgd

    # Standard deviation of the Maxwellian distribution
    tau = 1e-12 * dgd_std * sqrt(tensor(fiber_len / n_sec, dtype=float32,
                                        device=device))  # s/√km to s

    # Get phase
    phase = exp(tensor([[1j], [-1j]], dtype=cfloat, device=device) *
                omega * tau / 2)

    # Loop through the number of sections
    for n in range(n_sec):
        # Apply rotations
        Erot = Vh[n, ...] @ E_freq

        # Applying DGD
        Erot = phase * Erot

        # Apply rotations
        E_freq = U[n, ...] @ Erot

    return E_freq


def __cd(E_freq, dispersion, fiber_len, freq, sr, k, device):
    """
    Apply chromatic dispersion in the frequency domain.

    Args
    ----
        E_freq (torch.Tensor): Frequency-domain signal (n_ch, n_pol, n_s)
        dispersion (float): Dispersion coefficient [ps/nm/km]
        fiber_len (float): Fiber length [km]
        freq (torch.Tensor): Carrier frequencies per channel [Hz],
                             shape (n_ch,)
        sr (float): Symbol rate [Hz]
        k (int): Oversampling factor
        device (torch.device): Target device

    Returns
    -------
        torch.Tensor: Signal with chromatic dispersion applied
    """
    # Get the number of channels, polarizations, and samples
    n_ch, n_pol, n_s = E_freq.shape

    # Speed of light [m/s]
    c = 299792458

    # fiber length in meters
    L = fiber_len * 1e3

    # wavelengths [m], shape (n_ch, 1, 1)
    lambdas = (c / freq).view(-1, 1, 1)

    # Dispersion [s/m/m]
    disp = dispersion * 1e-12 / (1e-9 * 1e3)

    # Get the frequency vector
    omega = 2 * pi * k * sr * \
        arange(-1 / 2, 1 / 2, 1 / n_s, dtype=float32, device=device)

    # Calculating the CD frequency response
    G = exp(1j * ((disp * lambdas ** 2) / (4 * pi * c)) * L * omega ** 2)

    # Inserting CD to the transmitted signal
    E_freq = ifftshift(G * fftshift(E_freq))

    return E_freq


def __nonlinear_phase_shift(signal, fiber_len, gamma, device):
    """
    Apply nonlinear Kerr effect to an optical signal over a fiber span.

    Args
    ----
        signal (torch.Tensor): Complex signal of shape (1, n_pol, n_samples)
        fiber_len (float): Fiber length in km
        gamma (float): Nonlinear coefficient [1/W/km]
        device (torch.device): Target device (e.g., torch.device("cuda"))

    Returns
    -------
        torch.Tensor: Signal with nonlinear phase shift applied
    """
    # Total power per sample (summing across channels and polarizations)
    power = sum_torch(abs_torch(signal) ** 2, dim=(0, 1), keepdim=True)

    # Nonlinear phase rotation
    phi_nl = gamma * fiber_len * power  # shape: (n_ch, 1, n_samples)

    signal_out = signal * exp(1j * phi_nl)

    return signal_out


def __att_fiber(signal, fiber_len, att_db_km, device):
    """
    Apply linear attenuation to an optical signal over a given fiber length.

    This function simulates fiber attenuation using a specified attenuation
    coefficient in dB/km, converting it to a linear scale and applying
    exponential loss over the distance.

    Args
    ----
        signal (torch.Tensor): Complex input tensor of
                               shape (1, n_pol, n_samples)
        fiber_len (float): Fiber length in km
        att_db_km (float): Attenuation coefficient in dB per kilometer
        device (torch.device): Target device (e.g., torch.device("cuda"))

    Returns
    -------
        torch.Tensor: Attenuated signal, same shape as input
    """
    # Linear attenuation
    att_lin = att_db_km * log(tensor(10, dtype=float32, device=device)) / 10

    # Attenuated signal
    signal_out = signal * exp(-att_lin * fiber_len)

    return signal_out
