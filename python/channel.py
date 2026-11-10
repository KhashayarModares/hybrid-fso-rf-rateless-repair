"""Paper-derived Markov weather and conditional erasures on a common wall clock.

Chatzidiamantis et al., JSAC 2015, DOI 10.1109/JSAC.2015.2432512:
Section II-B/D, equations (5)-(15), Tables I/II, and Figure 8.
The optical-power sweep and rho=5 are this experiment's design settings.
No queue admission, Lyapunov power controller or average power cap is asserted.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf, gammainc, gammaln
from scipy.stats import ncx2

PHYSICAL = {
    'distance_m': 1400., 'optical_wavelength_m': 780e-9,
    'aperture_radius_m': .05, 'divergence_rad': .005,
    'sensitivity_w': 1e-6, 'rf_frequency_hz': 60e9, 'rf_power_w': .01,
    'antenna_gain_each_dbi': 44., 'oxygen_db_per_km': 15.1,
    'noise_dbm': -85., 'rf_snr_threshold_db': 24., 'rician_k_db': 6.,
    'symbol_energy_j': 1.,
}
# Table II: attenuation dB/km, rain dB/km, C_n^2 in m^(-2/3).
WEATHER = {'clear': (.43, 0., 5e-14), 'haze': (4.2, 0., 1.7e-14),
           'rain': (5.8, 5.6, 5e-15)}


def optical_parameters(weather):
    attenuation, _, cn2 = WEATHER[weather]
    p = PHYSICAL
    distance = p['distance_m']
    variance = 1.23 * cn2 * (2*math.pi/p['optical_wavelength_m'])**(7/6) * distance**(11/6)
    # Both exponents are 7/6 in the cited paper's equations (8),(9).
    alpha = 1/math.expm1(.49*variance/(1+1.11*variance**1.2)**(7/6))
    beta = 1/math.expm1(.51*variance/(1+.69*variance**1.2)**(7/6))
    geometry = erf(math.sqrt(math.pi)*p['aperture_radius_m'] /
                   (math.sqrt(2)*p['divergence_rad']*distance))**2
    path = geometry * 10**(-attenuation*distance/1000/10)
    return float(alpha), float(beta), float(path)


def gamma_gamma_cdf(z, alpha, beta):
    """CDF of X*Y, X~Gamma(alpha,1/alpha), Y~Gamma(beta,1/beta).

    Conditional integration evaluates the same distribution as equation (11),
    without a Meijer-G dependency. Returned quadrature error is checked.
    """
    if z <= 0:
        return 0.
    def integrand(x):
        if x == 0:
            return 0.
        density = math.exp(alpha*math.log(alpha)+(alpha-1)*math.log(x)-alpha*x-gammaln(alpha))
        return density * gammainc(beta, beta*z/x)
    value, error = quad(integrand, 0., math.inf, epsabs=1e-10, epsrel=1e-8, limit=250)
    if error > 2e-8 or not -1e-9 <= value <= 1+1e-9:
        raise ArithmeticError(f'Gamma-Gamma quadrature failed: value={value}, error={error}')
    return float(np.clip(value, 0., 1.))


def rf_erasure(weather):
    """Equation (15) through the equivalent noncentral chi-square CDF."""
    p = PHYSICAL
    km = p['distance_m']/1000
    wavelength = 299792458./p['rf_frequency_hz']
    path_db = (2*p['antenna_gain_each_dbi']
               - 20*math.log10(4*math.pi*p['distance_m']/wavelength)
               - (p['oxygen_db_per_km']+WEATHER[weather][1])*km)
    snr = p['rf_power_w']*10**(path_db/10)*p['symbol_energy_j']/10**((p['noise_dbm']-30)/10)
    k = 10**(p['rician_k_db']/10)
    threshold = 10**(p['rf_snr_threshold_db']/10)
    return float(ncx2.cdf(2*(1+k)*threshold/snr, 2, 2*k))


@lru_cache(maxsize=32)
def conditional_erasures(mean_fso, weather_pair):
    names = weather_pair.split('-')
    optics = [optical_parameters(name) for name in names]
    def probabilities(power):
        return tuple(gamma_gamma_cdf(PHYSICAL['sensitivity_w']/(h*power), a, b)
                     for a, b, h in optics)
    log_power = brentq(lambda lp: np.dot((2/3, 1/3), probabilities(math.exp(lp)))-mean_fso,
                      -20., 8., xtol=1e-11)
    power = math.exp(log_power)
    fso = probabilities(power)
    if abs(np.dot((2/3,1/3), fso)-mean_fso) > 1e-8:
        raise ArithmeticError('Stationary optical erasure calibration failed')
    return fso, tuple(rf_erasure(name) for name in names), power


def metadata(cfg):
    fso, rf, power = conditional_erasures(cfg.eps_fso, cfg.weather_pair)
    a, b = .1*cfg.transition_scale, .2*cfg.transition_scale
    return {
        'reference': 'Chatzidiamantis et al., JSAC 2015, Section II, Tables I/II, Fig. 8',
        'reference_doi': '10.1109/JSAC.2015.2432512',
        'weather_pair': cfg.weather_pair, 'transition_matrix': [[1-a,a],[b,1-b]],
        'stationary_distribution': [2/3,1/3], 'state_eigenvalue': 1-a-b,
        'epoch_s': cfg.epoch_rf_packets*cfg.durations[1],
        'conditional_fso_erasures': fso, 'conditional_rf_erasures': rf,
        'stationary_fso_erasure': float(np.dot((2/3,1/3),fso)),
        'stationary_rf_erasure': float(np.dot((2/3,1/3),rf)),
        'optical_power_w': power, 'physical_parameters': PHYSICAL,
        'weather_parameters': {name:WEATHER[name] for name in cfg.weather_pair.split('-')},
        'packet_state_convention': 'state at packet launch; no within-packet waveform model',
        'initialization': 'stationary, independently per block',
        'feedback': 'reliable reverse reports as in the protocol; no reverse erasures',
        'power_scope': 'optical power sets the requested mean erasure; no source-paper power-allocation constraint',
    }


@dataclass(frozen=True)
class ChannelBank:
    public_seed: int
    states: np.ndarray
    uniforms: np.ndarray
    probabilities: np.ndarray
    epoch_ticks: int
    fingerprint: str

    def state_at(self, launch_tick):
        epoch = int(launch_tick)//self.epoch_ticks
        if epoch < 0 or epoch >= len(self.states):
            raise ValueError('Channel wall-clock horizon exceeded')
        return int(self.states[epoch])

    def success(self, link, ordinal, launch_tick):
        state = self.state_at(launch_tick)
        return bool(self.uniforms[link, ordinal] >= self.probabilities[link, state])


def markov_states(rng, horizon, a, b):
    """Geometric dwell representation of the two-state chain, initialized at pi."""
    initial = int(rng.random() >= b/(a+b))
    labels = (initial+np.arange(horizon, dtype=np.int64)) % 2
    transitions = np.where(labels == 0, a, b)
    # Every dwell is at least one epoch; horizon dwells always cover horizon epochs.
    lengths = rng.geometric(transitions)
    return np.repeat(labels.astype(np.uint8), lengths)[:horizon]


def trial_inputs(cfg, seed):
    public, weather, noise = np.random.SeedSequence(seed).spawn(3)
    public_seed = int(np.random.default_rng(public).integers(0, 2**64, dtype=np.uint64))
    horizon = cfg.attempt_cap + math.ceil(cfg.attempt_cap/cfg.packet_bits) + 3
    states = markov_states(np.random.default_rng(weather), horizon,
                           .1*cfg.transition_scale, .2*cfg.transition_scale)
    uniforms = np.random.default_rng(noise).random((2, cfg.attempt_cap))
    fso, rf, _ = conditional_erasures(cfg.eps_fso, cfg.weather_pair)
    probabilities = np.asarray((fso,rf))
    epoch_ticks = int(cfg.packet_bits*cfg.rho*cfg.epoch_rf_packets)
    fingerprint = hashlib.sha256(states.tobytes()+uniforms.tobytes()
                                 +probabilities.tobytes()+public_seed.to_bytes(8,'little')).hexdigest()
    for array in (states, uniforms, probabilities):
        array.setflags(write=False)
    return ChannelBank(public_seed, states, uniforms, probabilities, epoch_ticks, fingerprint)
