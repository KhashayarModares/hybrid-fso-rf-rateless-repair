"""Outcome bitmaps and explicit finite-block report-position framing.

The ONLY receiver-to-transmitter report argument is a binary frame containing
a first-position field and W outcome bits. No graph, decoder state or receiver
list is attached. The graph IDs come from the transmitter's independent ledger.
Each digit is a logical on-air bit; Python string storage is not radio airtime.
"""
from collections import deque
from dataclasses import dataclass
import heapq
import math

EPS_TIME = 1e-15


@dataclass(frozen=True)
class TxEntry:
    end_s: float
    link: int
    ordinal: int
    graph_id: int


class TxLedger:
    """Packet association inferred exclusively from the transmitter's own log.

    Fixed-length packets, known link rates and equal known forward propagation
    delays make their completion order locally known. The ledger knows its own
    serialization completions, but never their outcomes. No shared RX list.
    Equal-time completions are ordered FSO before RF at BOTH endpoints.
    """
    def __init__(self, window):
        self.window = window
        self.scheduled = []
        self.completed = deque()
        self.count = [0, 0]
        self.last_end = [-math.inf, -math.inf]
        self.last_clock = -math.inf
        self.report_count = 0

    def launch(self, link, end_s, graph_id):
        if link not in (0, 1) or not math.isfinite(end_s) or end_s <= self.last_end[link]:
            raise ValueError('Invalid local forward schedule')
        ordinal = self.count[link]
        self.count[link] += 1
        self.last_end[link] = end_s
        entry = TxEntry(float(end_s), link, ordinal, int(graph_id))
        heapq.heappush(self.scheduled, (entry.end_s, link, ordinal, entry))
        return ordinal

    def advance(self, now):
        if not math.isfinite(now) or now < self.last_clock - EPS_TIME:
            raise ValueError('Transmitter clock moved backwards')
        self.last_clock = now
        while self.scheduled and self.scheduled[0][0] <= now + EPS_TIME:
            first = self.scheduled[0][0]
            batch = []
            while self.scheduled and self.scheduled[0][0] <= min(first, now) + EPS_TIME:
                batch.append(heapq.heappop(self.scheduled)[3])
            self.completed.extend(sorted(batch, key=lambda p: (p.link, p.ordinal)))

    def ready(self):
        return len(self.completed) >= self.window

    def next_ids(self):
        if not self.ready():
            raise ValueError('No complete locally known window')
        return tuple(list(self.completed)[:self.window])

    def accept(self, payload):
        if type(payload) is not str or len(payload) != self.window or set(payload) - {'0', '1'}:
            raise ValueError('Malformed bitmap; expected exactly W outcome bits')
        if not self.ready():
            raise ValueError('Feedback covers transmissions not yet completed')
        entries = tuple(self.completed.popleft() for _ in range(self.window))
        self.report_count += 1
        return tuple(e.graph_id for e, bit in zip(entries, payload) if bit == '1')

    def skip(self):
        """Drop the oldest window whose report was lost: its outcomes are never learned."""
        if not self.ready():
            raise ValueError('Lost report covers transmissions not yet completed')
        entries = tuple(self.completed.popleft() for _ in range(self.window))
        self.report_count += 1
        return entries


class RxWindow:
    """Receiver-local outcome order; an erased scheduled slot is still observed.

    Slot boundaries and link direction are assumed perfectly synchronized even
    when a whole packet/header is erased. This is NOT supplied by the codebook.
    """
    def __init__(self, window):
        self.window = window
        self.outcomes = deque()
        self.count = [0, 0]
        self.last_clock = -math.inf
        self.last_link = -1
        self.emitted = 0

    def observe(self, now, link, success):
        if link not in (0, 1) or not math.isfinite(now) or now < self.last_clock - EPS_TIME:
            raise ValueError('Invalid receiver slot order')
        if abs(now-self.last_clock) <= EPS_TIME and link <= self.last_link:
            raise ValueError('Equal-time slots must use FSO-before-RF ordering')
        ordinal = self.count[link]
        self.count[link] += 1
        self.outcomes.append('1' if success else '0')
        self.last_clock, self.last_link = now, link
        return ordinal

    def emit(self):
        if len(self.outcomes) < self.window:
            raise ValueError('Receiver has no complete bitmap window')
        self.emitted += self.window
        return ''.join(self.outcomes.popleft() for _ in range(self.window))

    def ready(self):
        return len(self.outcomes) >= self.window

    def emit_frame(self, attempt_cap):
        first = self.emitted
        return encode_report(first, self.emit(), attempt_cap)


def encode_report(first, bits, attempt_cap):
    """Fixed context, first-position header, W outcome bits; no Python objects on wire."""
    if (type(first) is not int or not 0 <= first < attempt_cap or type(bits) is not str
            or not bits or set(bits) - {'0', '1'} or first + len(bits) > attempt_cap):
        raise ValueError('Invalid report position or payload')
    return format(first, f'0{(attempt_cap-1).bit_length()}b') + bits


def report_position(frame, window, attempt_cap):
    """First completion position carried by a well-formed frame."""
    width = (attempt_cap-1).bit_length()
    if type(frame) is not str or len(frame) != width + window or set(frame) - {'0', '1'}:
        raise ValueError('Malformed feedback frame')
    return int(frame[:width], 2)


def decode_report(frame, window, attempt_cap, expected_first):
    width = (attempt_cap-1).bit_length()
    first = report_position(frame, window, attempt_cap)
    if first != expected_first or first + window > attempt_cap:
        raise ValueError('Stale, skipped, reordered or out-of-range bitmap position')
    return frame[width:]


@dataclass(frozen=True)
class FeedbackReport:
    frame: str
    snapshot_s: float
    start_s: float
    finish_s: float
    arrival_s: float
    usable_s: float
    lost: bool = False


class DedicatedFeedbackLink:
    """FIFO serialization independent of both forward links; ordered service.

    The one aggregate preparation/mirror-processing delay is lumped AFTER link
    arrival. It is a fixed latency, not a simulated CPU server or throughput.
    Snapshot preparation itself is instantaneous under this coding abstraction.
    A lost report is serialized like any other but never delivered.
    """
    def __init__(self, rate_bps, propagation_s, processing_s):
        # An infinite rate is allowed only for the ideal reference (zero serialization).
        if not rate_bps > 0:
            raise ValueError('Invalid feedback rate')
        if any(not math.isfinite(x) or x < 0 for x in (propagation_s, processing_s)):
            raise ValueError('Invalid feedback delay')
        self.rate_bps, self.propagation_s, self.processing_s = rate_bps, propagation_s, processing_s
        self.free_s = 0.
        self.last_snapshot_s = -math.inf
        self.pending = deque()
        self.reports = []

    def submit(self, frame, snapshot_s, lost=False):
        if (type(frame) is not str or not frame or set(frame) - {'0', '1'}
                or not math.isfinite(snapshot_s) or snapshot_s < self.last_snapshot_s):
            raise ValueError('Invalid feedback submission')
        start = max(snapshot_s, self.free_s)
        finish = start + len(frame) / self.rate_bps
        arrival = finish + self.propagation_s
        report = FeedbackReport(frame, snapshot_s, start, finish, arrival, arrival+self.processing_s,
                                bool(lost))
        if not lost:
            self.pending.append(report)
        self.reports.append(report)
        self.free_s, self.last_snapshot_s = finish, snapshot_s
        return report

    def next_usable(self):
        return self.pending[0].usable_s if self.pending else math.inf

    def deliver(self, now):
        ready = []
        while self.pending and self.pending[0].usable_s <= now + EPS_TIME:
            ready.append(self.pending.popleft())
        return ready

    def airtime_to(self, now):
        return sum(max(0., min(now, r.finish_s)-r.start_s) for r in self.reports)


class HalfDuplexFeedback:
    """Reports on the reverse RF link; forward RF and reports cannot overlap.

    A report starts at a TX opportunity t (W locally completed slots, forward RF
    serializer free, no report outstanding). The RX sends the frame once the last
    forward RF packet has arrived, t+d_f, and after the direction-change gap T (SIFS),
    at the RF rate R_R. Real: the TX keeps forward RF silent until the frame has
    reached it and a second gap T has passed, t+d_f+T+F/R_R+d_b+T (stop and wait), and
    uses it after the processing delay. A lost frame costs the same silence and is
    never delivered. Ideal: the report costs nothing; forward RF is not interrupted
    and the report is usable at t+d_f, when the RX has the outcomes.
    One report is outstanding at a time. FSO is never interrupted.
    """
    def __init__(self, rf_rate_bps, forward_propagation_s, feedback_propagation_s,
                 processing_s, ideal, turnaround_s=0.):
        if not math.isfinite(rf_rate_bps) or rf_rate_bps <= 0:
            raise ValueError('Invalid RF rate')
        if any(not math.isfinite(x) or x < 0 for x in
               (forward_propagation_s, feedback_propagation_s, processing_s, turnaround_s)):
            raise ValueError('Invalid feedback delay')
        self.rate_bps, self.ideal, self.turnaround_s = rf_rate_bps, ideal, turnaround_s
        self.forward_s, self.reverse_s, self.processing_s = (
            forward_propagation_s, feedback_propagation_s, processing_s)
        self.rf_free_s = 0.
        self.outstanding = None  # [start, snapshot, usable, frame or None]
        self.reports = []
        self.silenced = []

    def can_start(self, now):
        return self.outstanding is None and now >= self.rf_free_s - EPS_TIME

    def rf_blocked(self, now):
        return now < self.rf_free_s - EPS_TIME

    def start(self, now, frame_bits):
        snapshot = now + self.forward_s
        if self.ideal:
            usable = snapshot
        else:
            arrival = snapshot + self.turnaround_s + frame_bits / self.rate_bps + self.reverse_s
            self.rf_free_s = arrival + self.turnaround_s
            usable = arrival + self.processing_s
            self.silenced.append((now, self.rf_free_s))
        self.outstanding = [now, snapshot, usable, None, False]

    def snapshot_due(self, now):
        return (self.outstanding is not None and self.outstanding[3] is None
                and self.outstanding[1] <= now + EPS_TIME)

    def attach(self, frame, lost=False):
        start, snapshot, usable = self.outstanding[:3]
        self.outstanding[3], self.outstanding[4] = frame, bool(lost)
        airtime = 0. if self.ideal else len(frame) / self.rate_bps
        sent = snapshot + (0. if self.ideal else self.turnaround_s)
        self.reports.append(FeedbackReport(frame, snapshot, sent, sent + airtime,
                                           sent + airtime + (0. if self.ideal else self.reverse_s),
                                           usable, bool(lost)))

    def next_events(self, now):
        events = [self.rf_free_s] if self.rf_free_s > now + EPS_TIME else []
        if self.outstanding is not None:
            events.append(self.outstanding[1] if self.outstanding[3] is None else self.outstanding[2])
        return events

    def deliver(self, now):
        if (self.outstanding is not None and self.outstanding[3] is not None
                and self.outstanding[2] <= now + EPS_TIME):
            lost, self.outstanding = self.outstanding[4], None
            return [] if lost else [self.reports[-1]]
        return []

    def airtime_to(self, now):
        return sum(max(0., min(now, r.finish_s) - r.start_s) for r in self.reports)

    def silenced_to(self, now):
        return sum(max(0., min(now, b) - a) for a, b in self.silenced)
