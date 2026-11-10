"""Protocol, structural decoding and common-channel regression tests."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from coding import Config, PublicConfig, Codebook, Decoder, DEGREES, PMF
from bitmap_protocol import TxLedger, RxWindow
from model import Transmitter, Receiver, simulate, trial_inputs


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.public = self.cfg.public()

    def test_pmf_and_config(self):
        self.cfg.validate()
        self.assertAlmostEqual(float(PMF.sum()), 1.)
        self.assertTrue(np.all(PMF > 0))
        self.assertEqual(len(PMF), len(DEGREES))
        self.assertEqual(self.cfg.packet_bits, 1071)
        self.assertLessEqual(2*self.cfg.attempt_cap+self.cfg.k_hat, 2**self.cfg.index_bits)
        for bad in (replace(self.cfg, window=0), replace(self.cfg, window=12.5),
                    replace(self.cfg, eps_fso=1.1), replace(self.cfg, rho=0.),
                    replace(self.cfg, transition_scale=0),
                    replace(self.cfg, feedback_link='full_duplex'),
                    replace(self.cfg, state_threshold=1.5), replace(self.cfg, state_repair=1)):
            with self.assertRaises(ValueError):
                bad.validate()

    def test_graph_ids_independent_endpoints_random_order(self):
        for mapping in ('S0', 'S1'):
            sender = Codebook(self.public, 54321, mapping)
            receiver = Codebook(self.public, 54321, mapping)
            self.assertIsNot(sender, receiver)
            self.assertEqual(sender.precode, receiver.precode)
            n = self.cfg.attempt_cap
            ids = [0, n-1, n, 2*n-1, 2*n, 2*n+self.cfg.k_hat-1]
            ids += random.Random(44).sample(range(2*n+self.cfg.k_hat), 150)
            random.Random(55).shuffle(ids)
            for graph_id in ids:
                self.assertEqual(sender.neighbors(graph_id), receiver.read_index(sender.wire_index(graph_id)))
            with self.assertRaises(ValueError):
                receiver.read_index('1'*self.cfg.index_bits)

    def test_bitmap_association_without_receiver_objects(self):
        tx = TxLedger(4)
        rx = RxWindow(4)
        # Launch order differs from completion order; graph IDs are arbitrary.
        tx.launch(1, 5., 903)
        tx.launch(0, 1., 25)
        tx.launch(0, 2., 802)
        tx.launch(0, 5., 137)
        # Receiver supplies no graph IDs, sequence numbers or decoder objects.
        for t, link, ok in ((1.,0,True), (2.,0,False), (5.,0,True), (5.,1,True)):
            rx.observe(t, link, ok)
        tx.advance(5.)
        payload = rx.emit()
        self.assertEqual(payload, '1011')
        self.assertEqual(tx.accept(payload), (25, 137, 903))
        self.assertFalse(tx.ready())

    def test_mirror_exact_at_reported_prefix(self):
        book_tx = Codebook(self.public, 789)
        book_rx = Codebook(self.public, 789)
        tx = Transmitter(self.public, 'bitmap', book_tx)
        rx = Receiver(self.public, book_rx)
        n = self.cfg.attempt_cap
        ids = random.Random(31).sample(range(2*n+self.cfg.k_hat), 48)
        for i, graph_id in enumerate(ids):
            now = float(i+1)
            tx.ledger.launch(i%2, now, graph_id)
            tx.ledger.advance(now)
            bits = book_tx.wire_index(graph_id) if i%5 else None
            rx.receive(now, i%2, bits)
            if (i+1) % self.cfg.window == 0:
                payload = rx.windows.emit()
                self.assertIs(type(payload), str)
                tx.accept_bitmap(payload)
                self.assertEqual(tx.mirror.state(), rx.decoder.state())
                self.assertEqual(tx.mirror.eqs, rx.decoder.eqs)

    def test_no_receiver_or_channel_inputs_in_bitmap_controller(self):
        a = Transmitter(self.public, 'bitmap', Codebook(self.public, 123))
        b = Transmitter(self.public, 'bitmap', Codebook(self.public, 123))
        self.assertNotIn('eps_rf', vars(a.cfg))
        self.assertNotIn('eps_fso', vars(a.book.cfg))
        self.assertFalse(any(k in vars(a) for k in ('receiver','channel','outcomes','trial')))
        unrelated_receiver = Decoder(self.public, a.book.precode)
        for s in range(self.cfg.k):
            unrelated_receiver.add((s,))
        for i in range(6):
            self.assertEqual(a.launch(i%2, float(i)), b.launch(i%2, float(i)))
        self.assertFalse(hasattr(a, 'accept_ideal_state'))
        self.assertTrue(unrelated_receiver.done())
        with self.assertRaises(ValueError):
            Transmitter(self.cfg, 'bitmap', a.book)

    def test_bitmap_is_exactly_w_bits_and_rejects_objects(self):
        ledger = TxLedger(3)
        for i in range(3):
            ledger.launch(0, i+1., i)
        ledger.advance(3.)
        for invalid in ('01', '0100', '0x0', ['0','1','0'], {'bits':'010'}):
            with self.assertRaises(ValueError):
                ledger.accept(invalid)
        self.assertEqual(ledger.accept('000'), ())

    def test_prediction_is_not_confirmation(self):
        tx = Transmitter(self.public, 'bitmap', Codebook(self.public, 21))
        for _ in range(2):
            tx.mirror.add((0,1))
        tx.predicted = deepcopy(tx.mirror)
        before = tx.mirror.state()
        wire, _, _, targeted = tx.launch(1, 0.)
        self.assertTrue(targeted)
        self.assertEqual(tx.mirror.state(), before)
        self.assertGreater(tx.predicted.known.sum(), tx.mirror.known.sum())
        tx.ledger.advance(self.cfg.durations[1])
        # No actual success has been reported, despite the optimistic prediction.
        self.assertEqual(tx.mirror.received, 2)

    def test_decoder_peeling_and_symmetric_precode(self):
        cfg = replace(self.public, k=36, r=1)
        decoder = Decoder(cfg, ((0,1,2),))
        # Parity = x0+x1+x2. Known parity and two sources reveal the third.
        decoder.add((0,1))
        decoder.add((1,2))
        decoder.add((36,))
        decoder.add((0,))
        self.assertTrue(decoder.known[[0,1,2,36]].all())
        self.assertFalse(decoder.done())
        for i in range(3,36):
            decoder.add((i,))
        self.assertTrue(decoder.done())
        self.assertEqual(decoder.add((0,)), 0)

    def test_common_random_inputs_and_failure(self):
        a = trial_inputs(self.cfg, 987)
        b = trial_inputs(self.cfg, 987)
        self.assertEqual(a.public_seed, b.public_seed)
        self.assertEqual(a.fingerprint, b.fingerprint)
        np.testing.assert_array_equal(a.states, b.states)
        np.testing.assert_array_equal(a.uniforms, b.uniforms)
        bad = replace(self.cfg, attempt_multiplier=1)
        bank = replace(trial_inputs(bad,52), probabilities=np.ones((2,2)))
        result = simulate(bad, 'bitmap', 52, bank=bank)
        self.assertFalse(result['success'])
        self.assertIsNone(result['receiver_completion_s'])
        self.assertEqual(sum(result['received']), 0)

    def test_event_accounting_and_dedicated_feedback(self):
        result = simulate(self.cfg, 'bitmap', 80000000, record_trace=True)
        self.assertEqual(result['bitmap_payload_bits_launched'], self.cfg.window*result['bitmap_reports_launched'])
        self.assertLessEqual(result['bitmap_payload_bits_launched'], sum(result['completed']))
        self.assertEqual(result['launched'], [a+b for a,b in zip(result['completed'], result['in_flight'])])
        self.assertEqual(result['completed'], [a+b for a,b in zip(result['received'], result['erased'])])
        events = result['events']
        self.assertTrue(all(b['time_s'] >= a['time_s']-1e-15 for a,b in zip(events,events[1:])))
        fso = [e for e in events if e['event']=='data_launch' and e['link']==0]
        rf = [e for e in events if e['event']=='data_launch' and e['link']==1]
        feedback = [e for e in events if e['event']=='bitmap_snapshot']
        self.assertTrue(feedback)
        for launches in (fso,rf):
            for a,b in zip(launches,launches[1:]):
                self.assertAlmostEqual(a['completion_s'],b['time_s'],places=14)
            self.assertTrue(any(max(a['time_s'],b['start_s']) < min(a['completion_s'],b['finish_s'])
                                for a in launches for b in feedback))


if __name__ == '__main__':
    unittest.main()
