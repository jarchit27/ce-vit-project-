"""
channel_sim.py  —  Jake's/Clarke's model based TDL channel simulation for CE-ViT
Exact 3GPP TR 38.901 TDL-A/E tap powers and delays
Correct 5G NR pilot pattern (2 consecutive every 4 subcarriers)
Jake's isotropic scattering model (sum of sinusoids)
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 3GPP TR 38.901 TDL-A and TDL-E Tap Parameters (Table 7.7.2)
# ---------------------------------------------------------------------------

TDL_A = {
    'delays_ns': np.array([0, 30, 70, 90, 110, 190, 410]),
    'powers_db': np.array([-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9]),
}

TDL_E = {
    'delays_ns': np.array([0, 10, 20, 40, 80, 180]),
    'powers_db': np.array([-0.03, -22.03, -15.8, -18.1, -19.8, -22.9]),
}


# ---------------------------------------------------------------------------
# Jake's / Clarke's Doppler Spectrum (Sum of Sinusoids)
# ---------------------------------------------------------------------------

def jakes_doppler(
    max_doppler_hz: float,
    n_samples: int,
    T_eff: float,
    n_sinusoids: int = 20,
    rng: np.random.Generator = None,
):
    """
    Generate a time-varying complex fading process using Clarke's model.
    Uses sum of N sinusoids with uniformly spaced angles (Jake's method).

    Parameters
    ----------
    max_doppler_hz : maximum Doppler frequency (Hz)
    n_samples      : number of time samples (Nt OFDM symbols)
    T_eff          : effective OFDM symbol duration (s)
    n_sinusoids    : number of sinusoids (20 is standard)
    rng            : numpy random generator

    Returns
    -------
    h_doppler : (n_samples,) complex array — normalised Rayleigh fading process
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(n_samples) * T_eff
    h = np.zeros(n_samples, dtype=complex)

    for n in range(1, n_sinusoids + 1):
        # Uniformly spaced angles of arrival (Clarke's isotropic scattering)
        alpha_n = (2 * math.pi * n) / n_sinusoids
        f_d_n   = max_doppler_hz * math.cos(alpha_n)

        # Independent random initial phase per sinusoid
        phi_n = rng.uniform(-math.pi, math.pi)

        h += np.exp(1j * (2 * math.pi * f_d_n * t + phi_n))

    # Normalise to unit power
    h /= math.sqrt(n_sinusoids)
    return h


# ---------------------------------------------------------------------------
# TDL Channel Generator (Jake's model per tap)
# ---------------------------------------------------------------------------

def generate_tdl_channel(
    Nf: int = 120,
    Nt: int = 14,
    delay_spread_ns: float = 200.0,
    max_doppler_hz: float = 500.0,
    snr_db: float = 20.0,
    carrier_freq_hz: float = 3.5e9,
    subcarrier_spacing_hz: float = 15e3,
    channel: str = "TDL-A",
    n_sinusoids: int = 20,
    rng: np.random.Generator = None,
):
    """
    Generate a time-frequency channel matrix H ∈ C^{Nf x Nt} using
    the 3GPP TDL model with Jake's Doppler spectrum per tap.

    Parameters
    ----------
    Nf                  : number of subcarriers
    Nt                  : number of OFDM symbols
    delay_spread_ns     : RMS delay spread (ns) — scales tap delays
    max_doppler_hz      : maximum Doppler shift (Hz)
    snr_db              : signal-to-noise ratio (dB)
    carrier_freq_hz     : carrier frequency (Hz)
    subcarrier_spacing_hz : subcarrier spacing (Hz)
    channel             : 'TDL-A' or 'TDL-E'
    n_sinusoids         : number of sinusoids in Jake's model (default 20)
    rng                 : numpy random generator

    Returns
    -------
    H         : (Nf, Nt) complex — true channel matrix
    Y         : (Nf, Nt) complex — received signal
    noise_var : float
    """
    if rng is None:
        rng = np.random.default_rng()

    # Select tap parameters
    params = TDL_A if channel == "TDL-A" else TDL_E
    raw_delays_ns = params['delays_ns']
    powers_db     = params['powers_db']
    n_taps        = len(raw_delays_ns)

    # Linear tap powers, normalised to sum = 1
    powers = 10 ** (powers_db / 10)
    powers /= powers.sum()

    # Scale delays by delay_spread relative to 300 ns reference
    scaled_delays_ns = raw_delays_ns * (delay_spread_ns / 300.0)

    # Convert delays to fractional subcarrier bin indices
    T_sym   = 1.0 / subcarrier_spacing_hz        # OFDM symbol duration (s)
    delay_s = scaled_delays_ns * 1e-9
    delay_bin = delay_s / T_sym * Nf             # fractional bin indices

    # Effective symbol duration including CP (~7% overhead for 15kHz SCS)
    T_eff = 1.0 / subcarrier_spacing_hz * 1.07

    # Subcarrier index vector
    n_vec = np.arange(Nf)

    # Build channel matrix H (Nf x Nt)
    H = np.zeros((Nf, Nt), dtype=complex)

    for l in range(n_taps):
        tap_power = math.sqrt(powers[l])

        # Jake's model for this tap — time-varying complex amplitude
        h_tap_time = tap_power * jakes_doppler(
            max_doppler_hz=max_doppler_hz,
            n_samples=Nt,
            T_eff=T_eff,
            n_sinusoids=n_sinusoids,
            rng=rng,
        )  # shape (Nt,)

        # Frequency response of this tap: H_l(n) = h_l(t) * exp(-j2pi*n*tau_l/Nf)
        # Outer product: (Nf,) x (Nt,) → (Nf, Nt)
        freq_response = np.exp(-1j * 2 * math.pi * np.outer(n_vec, [delay_bin[l]]) / Nf)
        # freq_response shape: (Nf, 1), h_tap_time shape: (Nt,)
        H += freq_response * h_tap_time[np.newaxis, :]  # broadcast (Nf, Nt)

    # Normalise channel power
    H /= math.sqrt(np.mean(np.abs(H) ** 2))

    # Transmitted OFDM symbols (QPSK pilots = +1 for simplicity)
    X = np.ones((Nf, Nt), dtype=complex)

    # AWGN noise
    noise_var = 10 ** (-snr_db / 10.0)
    noise = math.sqrt(noise_var / 2.0) * (
        rng.standard_normal((Nf, Nt)) + 1j * rng.standard_normal((Nf, Nt))
    )

    Y = H * X + noise
    return H, Y, noise_var


# ---------------------------------------------------------------------------
# 5G NR Pilot Pattern (CORRECTED — 2 consecutive every 4 subcarriers)
# ---------------------------------------------------------------------------

def get_pilot_indices(Nf: int = 120, Nt: int = 14):
    """
    Correct 5G NR pilot pattern:
      Frequency : 2 consecutive pilots every 4 subcarriers
                  → indices 0,1, 4,5, 8,9, ... → Nf_p = 40 from Nf=120
      Time      : symbols 2 and 11 (0-indexed) → Nt_p = 2
    """
    freq_pilot_idx = np.array([i for i in range(Nf) if i % 4 in (0, 1)])[:40]
    time_pilot_idx = np.array([2, 11])
    return freq_pilot_idx, time_pilot_idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OFDMDataset(Dataset):
    """
    Pre-generated OFDM channel dataset using Jake's TDL model.

    Each sample:
      H_ls_p   : (Nf_p, Nt_p) complex — LS estimate at pilot positions
      H_true   : (Nf,   Nt  ) complex — true channel
      snr      : (1,) float  — normalised to [0, 1]
      doppler  : (1,) float  — normalised to [0, 1]
      delay    : (1,) float  — normalised to [0, 1]
    """

    # Parameter ranges matching paper Section IV
    DELAY_RANGE_NS   = list(range(25,  325, 25))   # 25,50,...,300 ns
    DOPPLER_RANGE_HZ = list(range(50, 1050, 50))   # 50,100,...,1000 Hz
    SNR_RANGE_DB     = list(range(0,   30,  5))    # 0,5,...,25 dB

    SNR_MAX     = 25.0
    DOPPLER_MAX = 1000.0
    DELAY_MAX   = 300.0

    def __init__(
        self,
        n_samples: int = 10240,
        Nf: int = 120,
        Nt: int = 14,
        channel: str = "TDL-A",
        fixed_snr: float = None,
        fixed_doppler: float = None,
        fixed_delay: float = None,
        seed: int = 42,
        n_sinusoids: int = 20,
    ):
        super().__init__()
        self.Nf, self.Nt = Nf, Nt
        rng = np.random.default_rng(seed)

        freq_idx, time_idx = get_pilot_indices(Nf, Nt)
        Nf_p = len(freq_idx)
        Nt_p = len(time_idx)

        H_ls_p_all = np.zeros((n_samples, Nf_p, Nt_p), dtype=complex)
        H_true_all = np.zeros((n_samples, Nf, Nt),     dtype=complex)
        snr_all     = np.zeros((n_samples, 1), dtype=np.float32)
        doppler_all = np.zeros((n_samples, 1), dtype=np.float32)
        delay_all   = np.zeros((n_samples, 1), dtype=np.float32)

        print(f"Generating {n_samples} Jake's TDL-{channel[-1]} channels ...")

        for i in range(n_samples):
            # Each sample gets its own independent random parameters
            snr_db = fixed_snr     if fixed_snr     is not None else float(rng.choice(self.SNR_RANGE_DB))
            dop_hz = fixed_doppler if fixed_doppler is not None else float(rng.choice(self.DOPPLER_RANGE_HZ))
            del_ns = fixed_delay   if fixed_delay   is not None else float(rng.choice(self.DELAY_RANGE_NS))

            H, Y, noise_var = generate_tdl_channel(
                Nf=Nf, Nt=Nt,
                delay_spread_ns=del_ns,
                max_doppler_hz=dop_hz,
                snr_db=snr_db,
                channel=channel,
                n_sinusoids=n_sinusoids,
                rng=rng,
            )

            # LS estimate at pilot positions
            H_p = H[np.ix_(freq_idx, time_idx)]
            Y_p = Y[np.ix_(freq_idx, time_idx)]
            X_p = np.ones_like(H_p)
            H_ls_p = Y_p / X_p       # LS: divide received by known pilot

            H_ls_p_all[i] = H_ls_p
            H_true_all[i] = H
            snr_all[i]     = snr_db / self.SNR_MAX
            doppler_all[i] = dop_hz / self.DOPPLER_MAX
            delay_all[i]   = del_ns / self.DELAY_MAX

            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{n_samples} samples generated")

        def to_complex_tensor(arr):
            return torch.view_as_complex(
                torch.from_numpy(
                    np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)
                )
            )

        self.H_ls_p  = to_complex_tensor(H_ls_p_all)
        self.H_true  = to_complex_tensor(H_true_all)
        self.snr     = torch.from_numpy(snr_all)
        self.doppler = torch.from_numpy(doppler_all)
        self.delay   = torch.from_numpy(delay_all)

    def __len__(self):
        return len(self.H_ls_p)

    def __getitem__(self, idx):
        return (
            self.H_ls_p[idx],
            self.H_true[idx],
            self.snr[idx],
            self.doppler[idx],
            self.delay[idx],
        )