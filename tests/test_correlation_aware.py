"""State-aware prediction/repair rules and half-duplex feedback timing."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from bitmap_protocol import HalfDuplexFeedback
from channel import trial_inputs
from coding import Config, Codebook, Decoder
from model import Transmitter, simulate


def transmitter(**kw):
    cfg = replace(Config(), window=4, **kw).public()
    return cfg, Transmitter(cfg,'bitmap',Codebook(cfg,4401))


def report(tx, links, bits, start=0.):
    """Launch one ordinary packet per link entry, complete them, deliver bits."""
    for i, link in enumerate(links):
        tx.ledger.launch(link, start+i+1., tx.book.ordinary_id(tx.launched, link))
        tx.launched += 1
    tx.ledger.advance(start+len(links))
    tx.accept_bitmap(bits)


class StateRuleTests(unittest.TestCase):
    def test_reported_state_uses_fso_entries_only_and_tau(self):
        _, tx = transmitter(state_threshold=.5)
        report(tx,(0,0,1,1),'1011')
        self.assertEqual((tx.fso_window_attempts,tx.fso_window_successes,tx.q_hat_fso),(2,1,.5))
        self.assertIs(tx.fso_good,True)          # equality is good
        report(tx,(0,0,0,1),'1001',10.)
        self.assertIs(tx.fso_good,False)
        report(tx,(1,1,1,1),'1111',20.)
        self.assertIsNone(tx.fso_good)          # no FSO attempt: unknown, not good/bad
        _, off = transmitter(state_threshold=0.)
        report(off,(0,0,0,0),'0000')
        self.assertIsNone(off.fso_good)          # tau=0 disables the state rule
        self.assertEqual(tx.bad_fso_windows,1)

    def test_bad_state_turns_rf_into_repairs_and_never_fso(self):
        _, tx = transmitter()
        report(tx,(0,0,0,0),'0000')
        self.assertIs(tx.fso_good,False)
        self.assertLess(tx.mirror.stagnation_run,tx.cfg.stall_trigger)
        _,_,_,targeted = tx.launch(0,10.)
        self.assertFalse(targeted)
        _,_,_,targeted = tx.launch(1,10.)
        self.assertTrue(targeted)
        self.assertEqual((tx.last_repair_reason,tx.state_launched),('state',1))
        _, no_m = transmitter(state_repair=False)
        report(no_m,(0,0,0,0),'0000')
        self.assertFalse(no_m.launch(1,10.)[3])
        _, good = transmitter()
        report(good,(0,0,0,0),'1111')
        good.launch(1,10.)  # may still repair through the unchanged stagnation rule
        self.assertNotEqual(good.last_repair_reason,'state')
        self.assertEqual(good.state_launched,0)

    def test_prediction_adds_rf_always_fso_only_when_good_and_never_confirms(self):
        for bits, fso_predicted in (('1111',True),('0000',False)):
            _, tx = transmitter(state_repair=False)
            report(tx,(0,0,0,0),bits)
            mirror = tx.mirror.state()
            self.assertEqual(tx.predicted.received,tx.mirror.received)
            tx.launch(0,10.)
            tx.launch(1,10.)
            self.assertEqual(tx.mirror.state(),mirror)
            self.assertEqual(tx.predicted.received-tx.mirror.received,1+int(fso_predicted))
            # Next report covers FSO@10, RF@10 and two more FSO slots. One completed FSO
            # slot and one RF packet still in serialization stay unreported.
            for t in (10.5,10.6,10.65):
                tx.ledger.launch(0,t,tx.book.ordinary_id(tx.launched,0)); tx.launched += 1
            tx.ledger.launch(1,10.8,tx.book.ordinary_id(tx.launched,1)); tx.launched += 1
            tx.ledger.advance(10.7)
            tx.accept_bitmap(bits)
            self.assertEqual((len(tx.ledger.completed),len(tx.ledger.scheduled)),(1,1))
            self.assertEqual(tx.predicted.received-tx.mirror.received,1+int(fso_predicted))

    def test_inflight_repairs_survive_a_report(self):
        _, tx = transmitter()
        report(tx,(0,0,0,0),'0000')
        _,_,end,targeted = tx.launch(1,10.)
        self.assertTrue(targeted)
        target = next(iter(tx.exclude))
        self.assertTrue(tx.predicted.known[target])
        self.assertGreater(end,10.+4e-6)
        # Next report covers four FSO slots that finish before the repair does.
        for i in range(4):
            tx.ledger.launch(0,10.+(i+1)*1e-6,tx.book.ordinary_id(tx.launched,0)); tx.launched += 1
        tx.ledger.advance(10.+4e-6)
        tx.accept_bitmap('0000')
        self.assertFalse(tx.mirror.known[target])
        self.assertTrue(tx.predicted.known[target])


PHYSICAL = dict(forward_propagation_s=1400/299792458., feedback_propagation_s=1400/299792458.,
                feedback_processing_s=1e-6)


class HalfDuplexTests(unittest.TestCase):
    def test_service_timing_real_and_ideal(self):
        real = HalfDuplexFeedback(200e6,4e-6,5e-6,1e-6,False)
        real.start(10e-6,25)
        self.assertTrue(real.rf_blocked(19e-6))
        self.assertFalse(real.rf_blocked(19.2e-6))
        self.assertAlmostEqual(real.rf_free_s,10e-6+4e-6+.125e-6+5e-6)
        self.assertFalse(real.snapshot_due(13.9e-6))
        self.assertTrue(real.snapshot_due(14e-6))
        real.attach('1'*25)
        self.assertEqual(real.deliver(real.rf_free_s),[])
        r = real.deliver(real.rf_free_s+1e-6)[0]
        self.assertAlmostEqual(r.usable_s-r.snapshot_s,.125e-6+5e-6+1e-6)
        ideal = HalfDuplexFeedback(200e6,4e-6,5e-6,1e-6,True)
        ideal.start(10e-6,25)
        self.assertFalse(ideal.rf_blocked(10.01e-6))      # no RF interruption
        self.assertAlmostEqual(ideal.outstanding[2],14e-6)  # usable once the RX has the outcomes
        self.assertFalse(ideal.can_start(10.2e-6))         # one report outstanding
        ideal.attach('1'*25)
        self.assertEqual(ideal.airtime_to(1.),0.)

    def test_physical_stop_and_wait_fso_continuity_and_exact_prefix(self):
        for ideal in (False,True):
            cfg = Config(window=8,feedback_link='half_duplex',feedback_ideal=ideal,**PHYSICAL)
            r = simulate(cfg,'bitmap',61207,True)
            self.assertTrue(r['success'])
            starts = [e for e in r['events'] if e['event']=='bitmap_start']
            self.assertTrue(starts)
            rf = [e for e in r['events'] if e['event']=='data_launch' and e['link']==1]
            fso = [e for e in r['events'] if e['event']=='data_launch' and e['link']==0]
            if ideal:
                self.assertEqual(r['rf_forward_silenced_s'],0.)
                self.assertEqual(r['bitmap_airtime_to_observation_s'],0.)
            else:
                per = (cfg.feedback_report_bits/cfg.rates[1]
                       + cfg.forward_propagation_s+cfg.feedback_propagation_s)
                for s in starts:
                    self.assertAlmostEqual(s['rf_free_s']-s['time_s'],per,places=14)
                    self.assertFalse(any(s['time_s']+1e-12 < e['time_s'] < s['rf_free_s']-1e-12 for e in rf))
                self.assertTrue(any(s['time_s'] < e['time_s'] < s['rf_free_s'] for s in starts for e in fso))
                self.assertAlmostEqual(r['rf_forward_silenced_s'],
                    sum(min(r['observation_end_s'],s['rf_free_s'])-s['time_s'] for s in starts))
            book = Codebook(cfg.public(),trial_inputs(cfg,61207).public_seed)
            oracle = Decoder(cfg.public(),book.precode)
            sent, done, snaps, k = {}, [], [], 0
            for e in r['events']:
                if e['event']=='data_launch': sent[e['link'],e['ordinal']] = e['index_bits']
                elif e['event']=='data_completion': done.append((e['link'],e['ordinal'],e['received']))
                elif e['event']=='bitmap_snapshot':
                    w = done[e['first_completion_position']:e['first_completion_position']+cfg.window]
                    self.assertEqual(e['payload'],''.join('1' if ok else '0' for *_,ok in w))
                    snaps.append(w)
                elif e['event']=='bitmap_arrival':
                    for link,ordinal,ok in snaps[k]:
                        if ok: oracle.add(book.read_index(sent[link,ordinal]))
                    k += 1
                    self.assertEqual(oracle.state(),e['mirror_state'])
            self.assertGreater(k,0)

    def test_default_feedback_cost_is_serialization_and_reverse_propagation(self):
        cfg = Config(feedback_link='half_duplex')
        d = 1400/299792458.
        self.assertEqual((cfg.forward_propagation_s,cfg.feedback_propagation_s,
                          cfg.feedback_processing_s),(d,d,0.))
        airtime = cfg.feedback_report_bits/cfg.rates[1]
        bank = trial_inputs(cfg,7310)
        real = simulate(cfg,'bitmap',7310,bank=bank)
        ideal = simulate(replace(cfg,feedback_ideal=True),'bitmap',7310,bank=bank)
        # Real half duplex: RF silent d_f + F/R_R + d_b per report; state age F/R_R + d_b.
        self.assertAlmostEqual(real['rf_forward_silenced_s'],
            real['bitmap_reports_launched']*(2*d+airtime),delta=2*d+airtime)
        self.assertAlmostEqual(real['feedback_mean_state_age_s'],airtime+d,places=12)
        self.assertEqual(ideal['rf_forward_silenced_s'],0.)
        self.assertEqual(ideal['feedback_mean_state_age_s'],0.)
        ded = simulate(replace(cfg,feedback_link='dedicated'),'bitmap',7310,bank=bank)
        self.assertGreaterEqual(ded['feedback_mean_state_age_s']+1e-15,
                                cfg.feedback_report_bits/cfg.feedback_rate_bps+d)
        for method in ('pooling','open_loop'):
            a = simulate(cfg,method,7310,bank=bank)
            b = simulate(replace(cfg,forward_propagation_s=0.),method,7310,bank=bank)
            self.assertAlmostEqual(a['receiver_completion_s']-b['receiver_completion_s'],d,places=15)

    def test_ideal_dedicated_reports_are_fresh(self):
        r = simulate(Config(feedback_ideal=True),'bitmap',9021)
        self.assertGreater(r['bitmap_reports_completed'],0)
        self.assertEqual(r['feedback_mean_state_age_s'],0.)
        self.assertEqual(r['bitmap_airtime_to_observation_s'],0.)


if __name__ == '__main__':
    unittest.main()
