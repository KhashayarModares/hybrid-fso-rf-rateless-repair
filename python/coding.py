"""Shared coding information and structural peeling; no channel state here.

RODD values and the ten degree-three precode checks are retained from the
supervisor implementation. This is not the standardized RaptorQ codec.
"""
from dataclasses import dataclass, fields
from functools import lru_cache
import math
import numpy as np

DEGREES = np.array(list(range(1, 31)) + [36], dtype=int)
PMF = np.array([
    .0764, .4407, .1469, .0734, .0441, .0294, .0210, .0157, .0122,
    .0098, .0080, .0067, .0056, .0048, .0042, .0037, .0032, .0029,
    .0026, .0023, .0021, .0019, .0017, .0016, .0015, .0014, .0013,
    .0012, .0011, .0010, .0675], dtype=float)
PMF /= PMF.sum()


@dataclass(frozen=True)
class PublicConfig:
    k: int = 128
    r: int = 10
    precode_degree: int = 3
    payload_bits: int = 1024
    crc_bits: int = 32
    fso_rate_bps: float = 1e9
    rho: float = 5.
    received_overhead_cap: float = 3.
    attempt_multiplier: int = 20
    window: int = 16
    stall_trigger: int = 2
    # Reported FSO state: good if q_hat_F >= tau. Zero disables the state rule.
    state_threshold: float = .5
    # Bad reported FSO state turns every RF opportunity into a targeted repair.
    state_repair: bool = True
    # 'dedicated': separate finite-rate link. 'half_duplex': reports use the RF link.
    feedback_link: str = 'dedicated'
    # Ideal feedback: the report costs nothing (no airtime, no RF interruption,
    # no serialization, reverse propagation or processing). It is usable as soon
    # as the RX has the reported outcomes and, for half duplex, an RF boundary.
    feedback_ideal: bool = False
    feedback_rate_bps: float = 10e6
    # System: 1400 m forward propagation for EVERY curve (pooling and open loop too).
    # Real feedback cost: serialization/airtime and 1400 m reverse propagation only;
    # no processing delay and no half-duplex turnaround guard.
    forward_propagation_s: float = 1400. / 299792458.
    feedback_propagation_s: float = 1400. / 299792458.
    feedback_processing_s: float = 0.
    # Real feedback only: each report is independently lost with this probability.
    # A lost report still occupies its airtime; its outcomes are never confirmed.
    feedback_loss_probability: float = 0.
    # Real half duplex only: direction-change gap, paid twice per report (receiver before
    # replying, transmitter before resuming). Used value: DMG aSIFSTime = 3 us, IEEE Std
    # 802.11-2024 Table 20-30; SIFS = aRxPHYDelay + aMACProcessingDelay + aRxTxTurnaroundTime.
    turnaround_s: float = 0.

    @property
    def feedback_header_bits(self):
        # First completion-position index in a fixed, pre-shared block context.
        return (self.attempt_cap - 1).bit_length()

    @property
    def feedback_report_bits(self):
        return self.feedback_header_bits + self.window

    @property
    def k_hat(self):
        return self.k + self.r

    @property
    def index_bits(self):
        return math.ceil(math.log2(256 * self.k))

    @property
    def packet_bits(self):
        return self.payload_bits + self.crc_bits + self.index_bits

    @property
    def received_cap(self):
        return math.ceil((1 + self.received_overhead_cap) * self.k)

    @property
    def attempt_cap(self):
        return min(self.attempt_multiplier * self.received_cap, 40 * self.k)

    @property
    def rates(self):
        return (self.fso_rate_bps, self.fso_rate_bps / self.rho)

    @property
    def durations(self):
        return tuple(self.packet_bits / r for r in self.rates)

    @property
    def slot_s(self):
        return self.packet_bits / sum(self.rates)

    def validate(self):
        for name in ('k', 'r', 'precode_degree', 'payload_bits', 'crc_bits',
                     'attempt_multiplier', 'window', 'stall_trigger'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.k <= 0 or self.payload_bits <= 0 or self.crc_bits < 0:
            raise ValueError('Invalid packet dimensions')
        if not all(math.isfinite(x) for x in (self.received_overhead_cap, self.rho,
                self.fso_rate_bps)):
            raise ValueError('Non-finite public parameter')
        if self.k_hat < int(DEGREES.max()) or not 1 <= self.precode_degree <= self.k:
            raise ValueError('Invalid source/precode size for this RODD support')
        if self.r < 0 or self.received_overhead_cap < 0 or self.attempt_multiplier < 1:
            raise ValueError('Invalid precode or numerical cap')
        if self.rho < 1 or not float(self.rho).is_integer() or self.fso_rate_bps <= 0:
            raise ValueError('This clock model requires positive rates and integer rho >= 1')
        if not 1 <= self.window <= self.attempt_cap or self.stall_trigger < 1:
            raise ValueError('Invalid report window or stall trigger')
        if (isinstance(self.state_threshold, bool)
                or not isinstance(self.state_threshold, (int, float))
                or not math.isfinite(self.state_threshold)
                or not 0 <= self.state_threshold <= 1):
            raise ValueError('State threshold must be in [0,1]; zero disables the state rule')
        if type(self.state_repair) is not bool or type(self.feedback_ideal) is not bool:
            raise ValueError('state_repair and feedback_ideal must be Boolean')
        if self.feedback_link not in ('dedicated', 'half_duplex'):
            raise ValueError('feedback_link must be dedicated or half_duplex')
        if 2 * self.attempt_cap + self.k_hat > 2 ** self.index_bits:
            raise ValueError('Ordinary and targeted codebook entries exceed index field')
        if not math.isfinite(self.feedback_rate_bps) or self.feedback_rate_bps <= 0:
            raise ValueError('Feedback service rate must be finite and positive')
        for name in ('forward_propagation_s', 'feedback_propagation_s', 'feedback_processing_s'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        p = self.feedback_loss_probability
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p < 1:
            raise ValueError('Feedback loss probability must lie in [0,1)')
        if p > 0 and self.feedback_ideal:
            raise ValueError('Ideal feedback has no report loss')
        t = self.turnaround_s
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError('Turnaround must be finite and nonnegative')
        if t > 0 and (self.feedback_link != 'half_duplex' or self.feedback_ideal):
            raise ValueError('Turnaround applies only to real half-duplex feedback')


@dataclass(frozen=True)
class Config(PublicConfig):
    """Experiment parameters. Erasure probabilities are not given to controllers."""
    eps_fso: float = .59
    weather_pair: str = 'clear-haze'
    transition_scale: float = 1.
    # Declared main scenario before publication runs: 10 RF packet durations
    # (53.55 us) per weather epoch, i.e. mean clear/haze dwell 536/268 us.
    epoch_rf_packets: int = 10

    def public(self):
        return PublicConfig(**{f.name: getattr(self, f.name) for f in fields(PublicConfig)})

    def validate(self):
        super().validate()
        if not 0 < self.eps_fso < 1:
            raise ValueError('The calibrated stationary FSO erasure must lie in (0,1)')
        if self.weather_pair not in ('clear-haze', 'clear-rain'):
            raise ValueError('Choose clear-haze or clear-rain from the paper state table')
        if not 0 < self.transition_scale <= 10/3:
            raise ValueError('Transition scale must lie in (0,10/3]')
        if not isinstance(self.epoch_rf_packets, int) or self.epoch_rf_packets < 1:
            raise ValueError('Channel epoch must be a positive integer number of RF packet durations')


def link_cdfs(rho, mapping):
    """Long-run ensemble allocation; S1 fills RF with degrees one then two."""
    share = 1 / (1 + rho)
    if mapping == 'S0':
        p = np.full(len(PMF), share)
    elif mapping == 'S1':
        p = np.zeros(len(PMF))
        budget = share
        for degree in (1, 2):
            i = int(np.flatnonzero(DEGREES == degree)[0])
            p[i] = min(1., budget / PMF[i])
            budget -= p[i] * PMF[i]
        if budget > 1e-12:
            remaining = 1 - p
            p += remaining * budget / np.dot(PMF, remaining)
    else:
        raise ValueError('Unknown mapping')
    assert abs(np.dot(PMF, p) - share) < 1e-10
    cdfs = [np.cumsum(PMF * (1-p) / (1-share)), np.cumsum(PMF * p / share)]
    for cdf in cdfs:
        cdf[-1] = 1.
    return tuple(cdfs)


class Codebook:
    """Public, block-specific graph generator. It never receives an erasure seed.

    IDs [0,N) and [N,2N) identify FSO/RF ordinary graphs; [2N,2N+k_hat)
    identify singleton targets. Ordinary graph IDs may be used in ANY order.
    N is the numerical attempt cap, not an assertion of infinite ID capacity.
    The session/block context and this generator are assumed pre-shared.
    """
    def __init__(self, cfg, public_seed, mapping='S0'):
        cfg = cfg.public() if isinstance(cfg, Config) else cfg
        self.cfg, self.public_seed = cfg, int(public_seed)
        self.cdfs = link_cdfs(cfg.rho, mapping)
        rng = self.rng(0, 0)
        self.precode = tuple(tuple(map(int, rng.choice(cfg.k, cfg.precode_degree, replace=False)))
                             for _ in range(cfg.r))
        self.transmission_order = self.rng(1, 0).permutation(cfg.attempt_cap)

    def rng(self, domain, index):
        return np.random.default_rng(np.random.SeedSequence(
            [self.public_seed & 0xffffffff, self.public_seed >> 32, domain, index]))

    def row(self, index):
        return _row(self.public_seed, self.cfg.k_hat, int(index))

    def ordinary_id(self, launch_number, link):
        return link * self.cfg.attempt_cap + int(self.transmission_order[launch_number])

    def target_id(self, symbol):
        if not 0 <= symbol < self.cfg.k_hat:
            raise ValueError('Invalid repair target')
        return 2 * self.cfg.attempt_cap + int(symbol)

    def neighbors(self, graph_id):
        n = self.cfg.attempt_cap
        if not 0 <= graph_id < 2*n + self.cfg.k_hat:
            raise ValueError('Invalid or reserved graph identifier')
        if graph_id >= 2*n:
            return (graph_id - 2*n,)
        link, row = divmod(graph_id, n)
        u, order = self.row(row)
        degree = int(DEGREES[np.searchsorted(self.cdfs[link], u, side='right')])
        return order[:degree]

    def wire_index(self, graph_id):
        self.neighbors(graph_id)  # reject reserved IDs, not just out-of-range bits
        return format(graph_id, f'0{self.cfg.index_bits}b')

    def read_index(self, bits):
        if type(bits) is not str or len(bits) != self.cfg.index_bits or set(bits) - {'0','1'}:
            raise ValueError('Malformed forward index field')
        return self.neighbors(int(bits, 2))


@lru_cache(maxsize=65536)
def _row(public_seed, k_hat, index):
    """Pure function of public block context: both endpoints compute identical rows.

    Caching it only avoids recomputing the same generator output; no state is shared.
    """
    rng = np.random.default_rng(np.random.SeedSequence(
        [public_seed & 0xffffffff, public_seed >> 32, 2, index]))
    u = float(rng.random())
    return u, tuple(map(int, rng.permutation(k_hat)))


class Decoder:
    """Structural peeling with symmetric precode checks, as in the supervisor code.

    No payload values, Gaussian elimination, processing delay or noisy decoding.
    'Progress' means newly recovered symbols, not algebraic rank innovation.
    """
    def __init__(self, cfg, precode):
        self.cfg, self.precode = cfg, tuple(tuple(p) for p in precode)
        self.known = np.zeros(cfg.k_hat, dtype=bool)
        self.covered = np.zeros(cfg.k_hat, dtype=bool)
        self.eqs = []
        self.sym2eqs = [[] for _ in range(cfg.k_hat)]
        self.q = []
        self.stagnation_run = self.received = 0
        self.n_known = self.n_known_sources = 0

    def copy(self):
        """Exact independent copy (equivalent to deepcopy, without its overhead)."""
        other = object.__new__(Decoder)
        other.cfg, other.precode = self.cfg, self.precode
        other.known, other.covered = self.known.copy(), self.covered.copy()
        other.eqs = [list(e) for e in self.eqs]
        other.sym2eqs = [list(e) for e in self.sym2eqs]
        other.q = list(self.q)
        other.stagnation_run, other.received = self.stagnation_run, self.received
        other.n_known, other.n_known_sources = self.n_known, self.n_known_sources
        return other

    def done(self):
        return self.n_known_sources == self.cfg.k

    def solve(self, symbol):
        if not self.known[symbol]:
            self.known[symbol] = True
            self.n_known += 1
            self.n_known_sources += int(symbol < self.cfg.k)
            self.q.append(symbol)

    def peel(self):
        while self.q:
            symbol = self.q.pop()
            for eid in self.sym2eqs[symbol]:
                self.eqs[eid] = [x for x in self.eqs[eid] if x != symbol]
                if len(self.eqs[eid]) == 1:
                    self.solve(self.eqs[eid][0])
            self.sym2eqs[symbol] = []

    def add(self, neighbors):
        neighbors = tuple(sorted(set(map(int, neighbors))))
        if not neighbors or min(neighbors) < 0 or max(neighbors) >= self.cfg.k_hat:
            raise ValueError('Invalid coding equation')
        before = self.n_known
        self.received += 1
        self.covered[list(neighbors)] = True
        reduced = [s for s in neighbors if not self.known[s]]
        eid = len(self.eqs)
        self.eqs.append(reduced)
        for symbol in reduced:
            self.sym2eqs[symbol].append(eid)
        if len(reduced) == 1:
            self.solve(reduced[0])
        self.peel()
        progress = True
        while progress:
            progress = False
            for j, sources in enumerate(self.precode):
                unknown = [s for s in (*sources, self.cfg.k+j) if not self.known[s]]
                if len(unknown) == 1:
                    self.solve(unknown[0])
                    self.peel()
                    progress = True
        gain = self.n_known - before
        self.stagnation_run = self.stagnation_run + 1 if not self.done() and gain == 0 else 0
        return gain

    def state(self):
        return {'received': self.received, 'known_ids': np.flatnonzero(self.known).tolist(),
                'unresolved_ids': np.flatnonzero(~self.known).tolist(),
                'uncovered_ids': np.flatnonzero(~self.known & ~self.covered).tolist(),
                'stagnation_run': self.stagnation_run}
