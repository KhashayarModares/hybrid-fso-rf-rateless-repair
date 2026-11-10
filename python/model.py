"""Receiver-completion simulation with isolated controller and outcome-only reports.

Stopping observation at receiver completion is a statistical measurement rule,
not an instantaneous STOP signal to the transmitter. Post-delivery transmission
and sender notification are outside this model; no STOP cost is claimed here.
"""
from dataclasses import dataclass
import math
import heapq
import numpy as np
from bitmap_protocol import (EPS_TIME, TxLedger, RxWindow, decode_report, report_position,
                             DedicatedFeedbackLink, HalfDuplexFeedback)
from coding import Config, PublicConfig, Codebook, Decoder
from channel import trial_inputs

METHODS = ('pooling', 'open_loop', 'bitmap')


class Transmitter:
    """Only public coding state, the local transmission log and delivered feedback.

    Ordinary choices and targeted choices do not read receiver/channel objects.
    """
    def __init__(self, cfg, method, book):
        if type(cfg) is not PublicConfig or method not in METHODS:
            raise ValueError('Transmitter requires public configuration and a known method')
        self.cfg, self.method, self.book = cfg, method, book
        self.ledger = TxLedger(cfg.window)
        self.mirror = Decoder(cfg, book.precode)
        self.predicted = self.mirror.copy()
        self.exclude = set()
        self.launched = 0
        self.q_hat_fso = None
        self.fso_good = None  # None until a report with FSO attempts arrives
        self.fso_window_attempts = self.fso_window_successes = 0
        self.fso_estimated_windows = self.bad_fso_windows = 0
        self.state_launched = 0
        self.unconfirmed = []  # launches whose report was lost: predicted, never confirmed
        self.last_repair_reason = 'ordinary'

    def update_fso_state(self, entries, bits):
        """Reported FSO state from the latest report only; no hidden channel state.

        q_hat_F is acknowledged FSO successes over FSO attempts in the report.
        The state is good when q_hat_F >= tau, bad otherwise, and unknown when the
        report has no FSO attempt or tau = 0 disables the state rule.
        """
        self.fso_window_attempts = sum(e.link == 0 for e in entries)
        self.fso_window_successes = sum(e.link == 0 and bit == '1'
                                        for e, bit in zip(entries, bits))
        self.q_hat_fso = (self.fso_window_successes / self.fso_window_attempts
                          if self.fso_window_attempts else None)
        self.fso_estimated_windows += int(self.q_hat_fso is not None)
        if self.q_hat_fso is None or self.cfg.state_threshold == 0:
            self.fso_good = None
        else:
            self.fso_good = self.q_hat_fso >= self.cfg.state_threshold
        self.bad_fso_windows += int(self.fso_good is False)

    def predict(self, link, graph_id):
        """Optimistically add one unreported launch to prediction, never to the mirror.

        Repairs and ordinary RF packets are predicted received. Ordinary FSO packets
        are predicted received only while the reported FSO state is good.
        """
        if graph_id >= 2 * self.cfg.attempt_cap or link == 1 or self.fso_good is True:
            self.predicted.add(self.book.neighbors(graph_id))

    def accept_bitmap(self, bits):
        if self.method != 'bitmap':
            raise ValueError('Bitmap input is disabled for this method')
        entries = self.ledger.next_ids()
        acknowledged = self.ledger.accept(bits)  # reject malformed input before state changes
        self.update_fso_state(entries, bits)
        for graph_id in acknowledged:
            self.mirror.add(self.book.neighbors(graph_id))
        self.predicted = self.mirror.copy()
        self.exclude.clear()
        # Launches not covered by any delivered report, in local completion order.
        unreported = list(self.ledger.completed) + sorted(
            (item[3] for item in self.ledger.scheduled), key=lambda e: (e.end_s, e.link))
        # Packets of lost reports keep the same prediction rule as unreported ones.
        for entry in unreported + self.unconfirmed:
            self.predict(entry.link, entry.graph_id)

    def accept_frame(self, frame):
        # Only the serialized frame crosses the endpoint boundary. Validate the
        # position header before changing the mirror, state or local cursor.
        expected = self.ledger.report_count * self.cfg.window
        if self.cfg.feedback_loss_probability > 0:
            # A later position reveals lost reports; their windows are skipped unlearned.
            first = report_position(frame, self.cfg.window, self.cfg.attempt_cap)
            gap = first - expected
            if gap < 0 or gap % self.cfg.window or len(self.ledger.completed) < gap + self.cfg.window:
                raise ValueError('Stale, reordered or not-yet-completed bitmap position')
            for _ in range(gap // self.cfg.window):
                self.unconfirmed.extend(self.ledger.skip())
            expected = first
        bits = decode_report(frame, self.cfg.window, self.cfg.attempt_cap, expected)
        self.accept_bitmap(bits)

    def select_target(self, order):
        d = self.predicted
        candidates = [s for s in order if not d.known[s] and s not in self.exclude]
        if not candidates:
            candidates = [s for s in order if not d.known[s]]
        if not candidates and self.unconfirmed:
            # Safety net after lost reports: the prediction may wrongly claim completion.
            # Prediction guides, confirmation guarantees: target what the mirror still lacks.
            d = self.mirror
            candidates = ([s for s in order if not d.known[s] and s not in self.exclude]
                          or [s for s in order if not d.known[s]])
        if not candidates:
            return None
        uncovered = [s for s in candidates if not d.covered[s]]
        if uncovered:
            return uncovered[0]
        # A coverage heuristic, not a claim that the precode cannot recover them.
        # Ties are resolved by the public pseudorandom order, not receiver state.
        return max(candidates, key=lambda s: sum(
            len(d.eqs[eid]) == 2 and s in d.eqs[eid] for eid in d.sym2eqs[s]))

    def launch(self, link, now):
        if self.launched >= self.cfg.attempt_cap:
            raise ValueError('Attempt cap reached')
        graph_id = self.book.ordinary_id(self.launched, link)
        targeted = False
        self.last_repair_reason = 'ordinary'
        stalled = self.mirror.stagnation_run >= self.cfg.stall_trigger
        bad_state = self.cfg.state_repair and self.fso_good is False
        # After lost reports, a prediction that claims completion while the confirmed
        # mirror does not is itself a reason to repair (safety net).
        unsure = bool(self.unconfirmed) and self.predicted.done()
        if (self.method == 'bitmap' and link == 1 and not self.mirror.done()
                and (stalled or bad_state or unsure)):
            row = int(self.book.transmission_order[self.launched])
            target = self.select_target(self.book.row(row)[1])
            if target is not None:
                graph_id = self.book.target_id(target)
                self.exclude.add(target)
                targeted = True
                self.last_repair_reason = ('stagnation' if stalled else 'state' if bad_state
                                           else 'safety')
                self.state_launched += int(not stalled and bad_state)
        if self.method == 'bitmap':
            # Explicit optimistic heuristic: no claim of successful delivery.
            self.predict(link, graph_id)
        end = now + self.cfg.durations[link]
        ordinal = self.ledger.launch(link, end, graph_id)
        self.launched += 1
        return self.book.wire_index(graph_id), ordinal, end, targeted


@dataclass(frozen=True)
class ChannelPacket:
    """Simulation-only forward frame. No neighbors/decoder state on this interface."""
    link: int
    ordinal: int
    index_bits: str
    start_s: float
    end_s: float
    success: bool
    targeted: bool  # observer label, not passed to Receiver.receive
    repair_reason: str = 'ordinary'  # observer label, not an on-air field


class Receiver:
    def __init__(self, cfg, book):
        self.book = book
        self.decoder = Decoder(cfg, book.precode)
        self.windows = RxWindow(cfg.window)

    def receive(self, now, link, index_bits):
        # An erasure carries no index or neighbors. Its scheduled slot is known.
        received = index_bits is not None
        self.windows.observe(now, link, received)
        return self.decoder.add(self.book.read_index(index_bits)) if received else 0


def feedback_service(cfg):
    if cfg.feedback_link == 'half_duplex':
        return HalfDuplexFeedback(cfg.rates[1], cfg.forward_propagation_s, cfg.feedback_propagation_s,
                                  cfg.feedback_processing_s, cfg.feedback_ideal, cfg.turnaround_s)
    if cfg.feedback_ideal:
        return DedicatedFeedbackLink(math.inf, 0., 0.)
    return DedicatedFeedbackLink(cfg.feedback_rate_bps, cfg.feedback_propagation_s,
                                 cfg.feedback_processing_s)


def simulate(cfg: Config, method: str, seed: int, record_trace=False, bank=None):
    cfg.validate()
    public = cfg.public()
    bank = trial_inputs(cfg, seed) if bank is None else bank
    public_seed, fingerprint = bank.public_seed, bank.fingerprint
    mapping = 'S1' if method == 'open_loop' else 'S0'
    # Independently initialized endpoints. Only the public block seed/config agree.
    tx = Transmitter(public, method, Codebook(public, public_seed, mapping))
    rx = Receiver(public, Codebook(public, public_seed, mapping))
    pending = [None, None]  # forward serializers, NOT propagation pipelines
    arrivals = []
    feedback = feedback_service(cfg)
    half_duplex = isinstance(feedback, HalfDuplexFeedback)
    # Feedback losses use their own stream: channel and graph randomness stay paired.
    loss_rng = (np.random.default_rng(np.random.SeedSequence([int(seed), 0x46424c]))
                if cfg.feedback_loss_probability > 0 else None)
    now = 0.
    launched = [0, 0]
    completed = [0, 0]
    received = [0, 0]
    data_airtime_allocated = [0., 0.]
    reports_completed = 0
    targeted_launched = targeted_received = redundant_repairs = 0
    state_received = state_productive = 0
    max_queue = 0
    events = [] if record_trace else None

    def log(kind, **fields):
        if events is not None:
            events.append({'time_s': now, 'event': kind, **fields})

    log('receiver_state', **rx.decoder.state())
    reason = None
    while True:
        # Transmitter predicts ONLY completion identities from its own local log.
        tx.ledger.advance(now)
        # Serialization completion frees a forward link regardless of feedback
        # or packets still propagating. RX outcome observations happen later.
        for link in (0, 1):
            packet = pending[link]
            if packet is not None and packet.end_s <= now + EPS_TIME:
                pending[link] = None
                log('data_serialized', link=link, ordinal=packet.ordinal)

        # Equal known forward propagation preserves TX completion order. The RX
        # sees a received index or an erased synchronized slot, not TX objects.
        due = []
        while arrivals and arrivals[0][0] <= now + EPS_TIME:
            due.append(heapq.heappop(arrivals)[3])
        for packet in sorted(due, key=lambda p: (p.link, p.ordinal)):
            link = packet.link
            gain = rx.receive(now, link, packet.index_bits if packet.success else None)
            completed[link] += 1
            received[link] += int(packet.success)
            if packet.targeted and packet.success:
                targeted_received += 1
                redundant_repairs += int(gain == 0)
                if packet.repair_reason == 'state':
                    state_received += 1
                    state_productive += int(gain > 0)
            log('data_completion', link=link, ordinal=packet.ordinal,
                received=packet.success, recovered_gain=gain, targeted=packet.targeted,
                repair_reason=packet.repair_reason)
            if packet.success and events is not None:  # trace-only snapshot
                log('receiver_state', **rx.decoder.state())

        if half_duplex and feedback.snapshot_due(now):
            # All W reported slots have arrived by t+d_f; the RX now sends them.
            first = rx.windows.emitted
            frame = rx.windows.emit_frame(cfg.attempt_cap)
            lost = loss_rng is not None and loss_rng.random() < cfg.feedback_loss_probability
            feedback.attach(frame, lost)
            report = feedback.reports[-1]
            log('bitmap_snapshot', payload=frame[cfg.feedback_header_bits:], wire_frame=frame,
                first_completion_position=first, start_s=report.start_s,
                finish_s=report.finish_s, arrival_s=report.arrival_s,
                usable_s=report.usable_s, queue_s=0., lost=lost)

        for report in feedback.deliver(now):
            tx.accept_frame(report.frame)
            reports_completed += 1
            if events is not None:  # trace-only snapshot
                log('bitmap_arrival', payload=report.frame[cfg.feedback_header_bits:],
                    wire_frame=report.frame, snapshot_s=report.snapshot_s,
                    usable_s=report.usable_s, state_age_s=now-report.snapshot_s,
                    mirror_state=tx.mirror.state(), predicted_state=tx.predicted.state(),
                    q_hat_fso=tx.q_hat_fso, fso_attempts=tx.fso_window_attempts,
                    fso_successes=tx.fso_window_successes, fso_good=tx.fso_good)

        if rx.decoder.done():
            reason = 'decoded'
            break
        if rx.decoder.received >= cfg.received_cap:
            reason = 'received_cap'
            break

        if method == 'bitmap' and not half_duplex:
            # Unsolicited fixed-W reports are generated at RX, without polls or any
            # inspection of the TX decoder/ledger. Only framed bits reach the TX.
            while rx.windows.ready():
                first = rx.windows.emitted
                frame = rx.windows.emit_frame(cfg.attempt_cap)
                lost = loss_rng is not None and loss_rng.random() < cfg.feedback_loss_probability
                report = feedback.submit(frame, now, lost)
                log('bitmap_snapshot', payload=frame[cfg.feedback_header_bits:],
                    wire_frame=frame, first_completion_position=first,
                    start_s=report.start_s, finish_s=report.finish_s,
                    arrival_s=report.arrival_s, usable_s=report.usable_s,
                    queue_s=report.start_s-now, lost=lost)
        elif (method == 'bitmap' and pending[1] is None and feedback.can_start(now)
              and tx.ledger.report_count*cfg.window + len(tx.ledger.completed)
              >= (len(feedback.reports) + 1)*cfg.window):
            # Public clocks and the TX ledger fix the opportunity; forward RF pauses.
            # The TX needs W completed slots beyond every report already started
            # (a lost report's slots stay in its ledger until the next report arrives).
            feedback.start(now, cfg.feedback_report_bits)
            log('bitmap_start', rf_free_s=feedback.rf_free_s, snapshot_s=feedback.outstanding[1],
                usable_s=feedback.outstanding[2])

        free = (pending[0] is None,
                pending[1] is None and not (half_duplex and feedback.rf_blocked(now)))
        for link in (0, 1):
            if not free[link] or tx.launched >= cfg.attempt_cap:
                continue
            wire, ordinal, end, targeted = tx.launch(link, now)
            launch_tick = round(now*cfg.fso_rate_bps)
            success = bank.success(link, ordinal, launch_tick)  # private PHY; common wall clock
            pending[link] = ChannelPacket(link, ordinal, wire, now, end, success, targeted,
                                           tx.last_repair_reason)
            heapq.heappush(arrivals, (end+cfg.forward_propagation_s, link, ordinal, pending[link]))
            launched[link] += 1
            targeted_launched += int(targeted)
            data_airtime_allocated[link] += cfg.durations[link]
            # Trace reveals a graph only after TX has selected it; no channel outcome.
            log('data_launch', link=link, ordinal=ordinal, index_bits=wire,
                targeted=targeted, repair_reason=tx.last_repair_reason,
                completion_s=end, receiver_arrival_s=end+cfg.forward_propagation_s,
                channel_epoch=launch_tick//bank.epoch_ticks,
                channel_state=bank.state_at(launch_tick))
        max_queue = max(max_queue, len(tx.ledger.completed))
        if tx.launched >= cfg.attempt_cap and not arrivals and all(p is None for p in pending):
            reason = 'attempt_cap'
            break
        feedback_events = (feedback.next_events(now) if half_duplex else [feedback.next_usable()])
        next_time = min([p.end_s for p in pending if p is not None]
                        + ([arrivals[0][0]] if arrivals else []) + feedback_events)
        if math.isinf(next_time):
            reason = 'attempt_cap'
            break
        if next_time < now-EPS_TIME:
            raise RuntimeError('Noncausal simulation clock')
        # Do not quantize propagation/feedback events onto the forward-bit clock.
        now = next_time

    success = reason == 'decoded'
    # Clip resource occupation at the observation endpoint. This is NOT latency
    # decomposed into a sum of overlapping forward and reverse airtimes.
    data_airtime = [data_airtime_allocated[i] - (
        max(0., pending[i].end_s-now) if pending[i] else 0.) for i in (0, 1)]
    started = [r for r in feedback.reports if r.start_s <= now + EPS_TIME]
    reports, bits_launched = len(started), cfg.window * len(started)
    reverse_allocated = sum(r.finish_s-r.start_s for r in started)
    reverse_airtime = feedback.airtime_to(now)
    rf_silenced = feedback.silenced_to(now) if half_duplex else 0.
    if max(*data_airtime, reverse_airtime) > now + 1e-12:
        raise RuntimeError('An independent link occupied more than observation time')
    if data_airtime[1] + rf_silenced > now + 1e-12:
        raise RuntimeError('Forward RF and half-duplex silence overlap')
    if bits_launched > sum(completed):
        raise RuntimeError('Feedback claims outcomes for uncompleted transmissions')
    log('observation_end', success=success, reason=reason)
    result = {'seed': seed, 'method': method, 'fingerprint': fingerprint,
              'success': success, 'termination': reason,
              'receiver_completion_s': now if success else None,
              'receiver_completion_slots': now/cfg.slot_s if success else None,
              'observation_end_s': now, 'launched': launched, 'completed': completed,
              'received': received, 'erased': [a-b for a,b in zip(completed,received)],
              'in_flight': [a-b for a,b in zip(launched,completed)],
              'bitmap_payload_bits_launched': bits_launched, 'bitmap_reports_launched': reports,
              'bitmap_reports_completed': reports_completed,
              'bitmap_airtime_to_observation_s': reverse_airtime,
              'bitmap_airtime_allocated_s': reverse_allocated,
              'bitmap_reports_generated': len(feedback.reports),
              'feedback_frame_bits_launched': cfg.feedback_report_bits * reports,
              'feedback_frame_bits_generated': cfg.feedback_report_bits * len(feedback.reports),
              'feedback_max_queue_s': max((r.start_s-r.snapshot_s for r in feedback.reports), default=0.),
              'feedback_mean_state_age_s': (sum(r.usable_s-r.snapshot_s for r in
                                               [r for r in feedback.reports if not r.lost][:reports_completed])
                                           / reports_completed if reports_completed else 0.),
              'feedback_reports_lost': sum(r.lost for r in started),
              'rf_forward_silenced_s': rf_silenced,
              'data_airtime_to_observation_s': data_airtime,
              'targeted_launched': targeted_launched, 'targeted_received': targeted_received,
              'received_targeted_without_recovery': redundant_repairs,
              'fso_estimated_windows': tx.fso_estimated_windows,
              'bad_fso_windows': tx.bad_fso_windows,
              'state_launched': tx.state_launched,
              'state_received': state_received,
              'state_productive': state_productive,
              'receiver_state': rx.decoder.state(), 'max_tx_completed_queue': max_queue,
              'feedback_extra_framing_bits_modeled': cfg.feedback_header_bits,
              'feedback_architecture': f"{cfg.feedback_link}{' ideal' if cfg.feedback_ideal else ''}",
              'timing_endpoint': 'receiver decoding completion; no terminal STOP model'}
    if events is not None:
        result['events'] = events
    return result
