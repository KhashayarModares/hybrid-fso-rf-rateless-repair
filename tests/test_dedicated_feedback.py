"""Independent framing, causal delay, FIFO and parallel-resource tests."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from bitmap_protocol import encode_report, decode_report, DedicatedFeedbackLink
from coding import Config, Codebook
from channel import trial_inputs
from model import simulate, Transmitter


class DedicatedTests(unittest.TestCase):
    def test_header_width_and_strict_wire_roundtrip(self):
        cfg = Config()
        self.assertEqual(cfg.feedback_header_bits,13)
        frame = encode_report(24,'101001010101',cfg.attempt_cap)
        self.assertEqual(len(frame),25)
        self.assertEqual(decode_report(frame,12,cfg.attempt_cap,24),'101001010101')
        for first in (0,12,25):
            with self.assertRaises(ValueError): decode_report(frame,12,cfg.attempt_cap,first)
        for bad in (frame[:-1],frame+'0',frame[:-1]+'x',{'frame':frame}):
            with self.assertRaises(ValueError): decode_report(bad,12,cfg.attempt_cap,24)
        with self.assertRaises(ValueError): encode_report(5119,'11',cfg.attempt_cap)

    def test_fifo_queue_and_exact_usable_time(self):
        link = DedicatedFeedbackLink(1e6,5e-6,2e-6)
        a = link.submit('1'*24,0.)
        b = link.submit('0'*24,10e-6)
        self.assertAlmostEqual(a.usable_s,31e-6)
        self.assertAlmostEqual(b.start_s-b.snapshot_s,14e-6)
        self.assertAlmostEqual(b.usable_s,55e-6)
        self.assertEqual(link.deliver(30e-6),[])
        self.assertEqual(link.deliver(31e-6),[a])
        self.assertEqual(link.deliver(55e-6),[b])
        self.assertAlmostEqual(link.airtime_to(30e-6),30e-6)
        with self.assertRaises(ValueError): link.submit('1',9e-6)

    def test_frame_validation_precedes_controller_mutation(self):
        cfg = Config(window=4).public()
        tx = Transmitter(cfg,'bitmap',Codebook(cfg,44))
        for i in range(4): tx.ledger.launch(0,i+1.,i)
        tx.ledger.advance(4.)
        before = tx.mirror.state()
        with self.assertRaises(ValueError): tx.accept_frame(encode_report(4,'1111',cfg.attempt_cap))
        self.assertEqual(tx.mirror.state(),before)
        self.assertEqual(tx.ledger.report_count,0)
        tx.accept_frame(encode_report(0,'0000',cfg.attempt_cap))
        self.assertEqual(tx.ledger.report_count,1)

    def test_propagation_applies_equally_to_both_baselines(self):
        base = Config(forward_propagation_s=0.)
        delayed = replace(base,forward_propagation_s=7e-6)
        bank = trial_inputs(base,5300)
        for method in ('pooling','open_loop'):
            a = simulate(base,method,5300,bank=bank)
            b = simulate(delayed,method,5300,bank=bank)
            self.assertAlmostEqual(b['receiver_completion_s']-a['receiver_completion_s'],7e-6)
            self.assertEqual(a['received'],b['received'])

    def test_every_report_obeys_timing_and_reconstructs_only_its_prefix(self):
        cfg = Config(window=8,feedback_rate_bps=1e6,state_threshold=.5)
        r = simulate(cfg,'bitmap',18721,True)
        completions, snapshots, acknowledged = [], [], []
        book = Codebook(cfg.public(),trial_inputs(cfg,18721).public_seed)
        from coding import Decoder
        oracle = Decoder(cfg.public(),book.precode)
        sent = {}
        for e in r['events']:
            if e['event']=='data_launch': sent[e['link'],e['ordinal']] = e['index_bits']
            elif e['event']=='data_completion':
                completions.append((e['link'],e['ordinal'],e['received']))
            elif e['event']=='bitmap_snapshot':
                first = e['first_completion_position']
                window = completions[first:first+cfg.window]
                self.assertEqual(e['payload'],''.join('1' if ok else '0' for _,_,ok in window))
                self.assertAlmostEqual(e['usable_s'],e['time_s']+e['queue_s']+
                    len(e['wire_frame'])/cfg.feedback_rate_bps+cfg.feedback_propagation_s+
                    cfg.feedback_processing_s,places=14)
                snapshots.append((e,window))
            elif e['event']=='bitmap_arrival':
                snap,window = snapshots[len(acknowledged)]
                self.assertAlmostEqual(e['time_s'],snap['usable_s'],places=14)
                for link,ordinal,ok in window:
                    if ok: oracle.add(book.read_index(sent[link,ordinal]))
                self.assertEqual(oracle.state(),e['mirror_state'])
                acknowledged.append(e)
        self.assertTrue(acknowledged)
        self.assertGreater(r['feedback_max_queue_s'],0)
        self.assertEqual(r['feedback_frame_bits_launched'],cfg.feedback_report_bits*r['bitmap_reports_launched'])
        self.assertLessEqual(r['bitmap_reports_generated'],sum(r['completed'])//cfg.window)
        self.assertEqual(r['receiver_completion_s'],r['events'][-1]['time_s'])

    def test_no_feedback_parameter_leaks_into_channel_or_nonfeedback_policies(self):
        a = Config()
        b = replace(a,feedback_rate_bps=1e5,feedback_propagation_s=.01,feedback_processing_s=.02)
        bank = trial_inputs(a,2718)
        self.assertEqual(bank.fingerprint,trial_inputs(b,2718).fingerprint)
        for other in (b, replace(a,feedback_link='half_duplex'), replace(a,feedback_ideal=True)):
            for method in ('pooling','open_loop'):
                ra,rb = simulate(a,method,2718,bank=bank),simulate(other,method,2718,bank=bank)
                ra.pop('feedback_architecture'); rb.pop('feedback_architecture')
                self.assertEqual(ra,rb)
                self.assertEqual(rb['feedback_frame_bits_launched'],0)

    def test_all_erasure_failure_and_slow_feedback_terminate_without_waiting_for_useless_feedback(self):
        cfg = Config(feedback_rate_bps=1.,attempt_multiplier=1)
        bank = replace(trial_inputs(cfg,341),probabilities=np.ones((2,2)))
        r = simulate(cfg,'bitmap',341,bank=bank)
        self.assertEqual(r['termination'],'attempt_cap')
        self.assertEqual(sum(r['completed']),cfg.attempt_cap)
        self.assertLess(r['observation_end_s'],1.)
        self.assertEqual(r['bitmap_reports_completed'],0)

    def test_invalid_feedback_parameters(self):
        for kw in ({'feedback_rate_bps':0},{'feedback_rate_bps':float('inf')},
                   {'feedback_processing_s':-1},{'forward_propagation_s':float('nan')},
                   {'feedback_propagation_s':-1}):
            with self.assertRaises(ValueError): replace(Config(),**kw).validate()

    def test_unusable_feedback_cannot_provide_secret_receiver_information(self):
        cfg = Config(feedback_processing_s=1.,state_threshold=.5)
        bank = trial_inputs(cfg,77901)
        pooling = simulate(cfg,'pooling',77901,bank=bank)
        bitmap = simulate(cfg,'bitmap',77901,bank=bank)
        self.assertEqual(bitmap['bitmap_reports_completed'],0)
        self.assertEqual(bitmap['targeted_launched'],0)
        self.assertEqual(bitmap['receiver_completion_s'],pooling['receiver_completion_s'])
        self.assertEqual(bitmap['received'],pooling['received'])


if __name__=='__main__': unittest.main()
