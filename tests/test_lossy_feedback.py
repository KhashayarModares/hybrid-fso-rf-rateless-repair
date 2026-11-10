"""Independent report loss and processing delay on the real dedicated link."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from bitmap_protocol import DedicatedFeedbackLink, encode_report
from channel import trial_inputs
from coding import Config, Codebook, Decoder
from model import Transmitter, simulate


def lossy(p=.3, **kw):
    return replace(Config(), feedback_loss_probability=p, feedback_processing_s=1e-6, **kw)


class LossyFeedbackTests(unittest.TestCase):
    def test_only_real_feedback_can_lose_reports_and_only_half_duplex_turns_around(self):
        with self.assertRaises(ValueError): lossy(feedback_ideal=True).validate()
        for p in (-.1, 1., float('nan'), True):
            with self.assertRaises(ValueError): replace(Config(),feedback_loss_probability=p).validate()
        lossy().validate()
        lossy(feedback_link='half_duplex', turnaround_s=2e-6).validate()
        for kw in ({}, {'feedback_link':'half_duplex','feedback_ideal':True}):
            with self.assertRaises(ValueError): replace(Config(),turnaround_s=2e-6,**kw).validate()
        with self.assertRaises(ValueError): replace(Config(),feedback_link='half_duplex',turnaround_s=-1.).validate()

    def test_half_duplex_turnaround_and_loss_timing(self):
        cfg = lossy(.3, feedback_link='half_duplex', turnaround_s=2e-6, window=16)
        r = simulate(cfg,'bitmap',61207,True)
        self.assertTrue(r['success'])
        d, T = cfg.forward_propagation_s, cfg.turnaround_s
        per = d + T + cfg.feedback_report_bits/cfg.rates[1] + cfg.feedback_propagation_s + T
        starts = [e for e in r['events'] if e['event']=='bitmap_start']
        for s in starts:
            self.assertAlmostEqual(s['rf_free_s']-s['time_s'],per,places=14)
        snaps = [e for e in r['events'] if e['event']=='bitmap_snapshot']
        self.assertTrue(any(e['lost'] for e in snaps) and not all(e['lost'] for e in snaps))
        arrivals = [e for e in r['events'] if e['event']=='bitmap_arrival']
        self.assertEqual(len(arrivals),sum(not e['lost'] for e in snaps if e['usable_s'] <= r['observation_end_s']))
        for e in snaps:   # RX starts sending after forward arrival plus its own turnaround
            self.assertAlmostEqual(e['start_s']-e['time_s'],T,places=14)

    def test_run_script_uses_model_rf_outage_sifs_and_rf_rate(self):
        import re
        from channel import rf_erasure
        script = (Path(__file__).resolve().parents[1]/'run_replacement.sh').read_text()
        default = lambda name: re.search(rf'{name}="\$\{{{name}:-([^}}]+)\}}"', script).group(1)
        self.assertEqual(float(default('LOSS')), rf_erasure('clear'))     # source: channel model
        self.assertEqual(rf_erasure('clear'), rf_erasure('haze'))
        self.assertEqual(float(default('TURN_US')), 3.)                   # DMG aSIFSTime, 802.11-2024 T.20-30
        self.assertEqual(float(default('FB_RATE_MBPS'))*1e6, Config().rates[1])  # RF link rate

    def test_safety_net_repairs_what_the_mirror_lacks_after_a_loss(self):
        cfg = lossy(window=4).public()
        tx = Transmitter(cfg,'bitmap',Codebook(cfg,44))
        for s in range(cfg.k_hat):    # prediction wrongly believes everything is decoded
            tx.predicted.add((s,))
        self.assertTrue(tx.predicted.done() and not tx.mirror.done())
        self.assertFalse(tx.launch(1,0.)[3])          # no lost report yet: no safety net
        tx.unconfirmed.append(None)                    # a report has been lost
        _,_,_,targeted = tx.launch(1,10.)
        self.assertTrue(targeted)
        self.assertEqual(tx.last_repair_reason,'safety')
        self.assertFalse(tx.mirror.known[next(iter(tx.exclude))])

    def test_lost_report_uses_airtime_but_is_never_delivered(self):
        link = DedicatedFeedbackLink(1e6,5e-6,1e-6)
        a = link.submit('1'*20,0.,lost=True)
        b = link.submit('0'*20,1e-6)
        self.assertAlmostEqual(b.start_s,a.finish_s)          # the lost frame still occupied the link
        self.assertEqual(link.deliver(1.),[b])
        self.assertAlmostEqual(link.airtime_to(1.),40e-6)

    def test_gap_skips_lost_window_and_rejects_stale_or_premature_positions(self):
        cfg = lossy(window=4).public()
        tx = Transmitter(cfg,'bitmap',Codebook(cfg,44))
        for i in range(12): tx.ledger.launch(0,i+1.,i)
        tx.ledger.advance(12.)
        before = tx.mirror.state()
        for first in (12, 2):   # beyond completed slots; not window aligned
            with self.assertRaises(ValueError): tx.accept_frame(encode_report(first,'1111',cfg.attempt_cap))
        self.assertEqual((tx.mirror.state(),tx.ledger.report_count),(before,0))
        tx.accept_frame(encode_report(4,'0000',cfg.attempt_cap))     # report for slots 0-3 was lost
        self.assertEqual((tx.ledger.report_count,len(tx.ledger.completed)),(2,4))
        # Lost packets stay predicted under the unreported rule; the mirror never confirms them.
        self.assertEqual([e.graph_id for e in tx.unconfirmed],[0,1,2,3])
        self.assertEqual(tx.mirror.received,0)
        good = Transmitter(cfg,'bitmap',Codebook(cfg,44))
        for i in range(12): good.ledger.launch(0,i+1.,i)
        good.ledger.advance(12.)
        good.accept_frame(encode_report(4,'1111',cfg.attempt_cap))   # FSO good: lost FSO predicted
        self.assertEqual(good.predicted.received-good.mirror.received,4+4)
        self.assertEqual(tx.predicted.received-tx.mirror.received,0)     # FSO bad: not predicted
        with self.assertRaises(ValueError): tx.accept_frame(encode_report(4,'0000',cfg.attempt_cap))
        lossless = Transmitter(Config(window=4).public(),'bitmap',Codebook(cfg,44))
        for i in range(8): lossless.ledger.launch(0,i+1.,i)
        lossless.ledger.advance(8.)
        with self.assertRaises(ValueError): lossless.accept_frame(encode_report(4,'0000',cfg.attempt_cap))

    def test_mirror_learns_only_delivered_reports_with_processing_delay(self):
        cfg = lossy(window=8)
        r = simulate(cfg,'bitmap',51207,True)
        self.assertTrue(r['success'])
        book = Codebook(cfg.public(),trial_inputs(cfg,51207).public_seed)
        oracle = Decoder(cfg.public(),book.precode)
        sent, done, snaps = {}, [], []
        for e in r['events']:
            if e['event']=='data_launch': sent[e['link'],e['ordinal']] = e['index_bits']
            elif e['event']=='data_completion': done.append((e['link'],e['ordinal'],e['received']))
            elif e['event']=='bitmap_snapshot':
                w = done[e['first_completion_position']:e['first_completion_position']+cfg.window]
                self.assertEqual(e['payload'],''.join('1' if ok else '0' for *_,ok in w))
                if not e['lost']:
                    snaps.append((e,w))
            elif e['event']=='bitmap_arrival':
                snap,w = snaps.pop(0)
                self.assertAlmostEqual(e['time_s']-snap['arrival_s'],cfg.feedback_processing_s,places=14)
                for link,ordinal,ok in w:
                    if ok: oracle.add(book.read_index(sent[link,ordinal]))
                self.assertEqual(oracle.state(),e['mirror_state'])
        self.assertGreater(r['feedback_reports_lost'],0)
        self.assertEqual(r['bitmap_reports_completed']+r['feedback_reports_lost']+len(snaps),
                         r['bitmap_reports_launched'])

    def test_loss_stream_keeps_channel_pairing_and_is_reproducible(self):
        a, b = Config(), lossy()
        bank = trial_inputs(a,2718)
        self.assertEqual(bank.fingerprint,trial_inputs(b,2718).fingerprint)
        for method in ('pooling','open_loop'):
            ra, rb = simulate(a,method,2718,bank=bank), simulate(b,method,2718,bank=bank)
            self.assertEqual(ra['receiver_completion_s'],rb['receiver_completion_s'])
        self.assertEqual(simulate(b,'bitmap',2718),simulate(b,'bitmap',2718))
        lost = sum(simulate(lossy(.085),'bitmap',s)['feedback_reports_lost'] for s in range(40))
        sent = sum(simulate(lossy(.085),'bitmap',s)['bitmap_reports_launched'] for s in range(40))
        self.assertLess(abs(lost/sent-.085),.035)


if __name__=='__main__': unittest.main()
