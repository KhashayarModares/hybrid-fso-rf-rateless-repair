"""Physical CDF, Markov statistics, real-time coupling and clock checks."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from channel import (conditional_erasures, gamma_gamma_cdf, optical_parameters,
                     rf_erasure, markov_states, trial_inputs, metadata)
from coding import Config
from model import simulate, METHODS


class ChannelTests(unittest.TestCase):
    def test_paper_physical_probabilities(self):
        # Equations (5)-(15), Tables I/II at the paper's 40 mW optical power.
        observed = []
        for name in ('clear','haze'):
            a,b,h = optical_parameters(name)
            observed.append(gamma_gamma_cdf(1e-6/(.04*h),a,b))
        np.testing.assert_allclose(observed,[.13947823296666,.60135497854819],rtol=1e-8)
        self.assertAlmostEqual(rf_erasure('clear'), .04868957933020315, places=10)
        self.assertEqual(rf_erasure('clear'),rf_erasure('haze'))
        self.assertGreater(rf_erasure('rain'),rf_erasure('clear'))

    def test_gamma_product_and_symmetry(self):
        a,b = 4.3,2.6
        rng = np.random.default_rng(8091)
        samples = rng.gamma(a,1/a,250000)*rng.gamma(b,1/b,250000)
        for threshold in (.2,.7,1.5):
            exact = gamma_gamma_cdf(threshold,a,b)
            self.assertAlmostEqual(exact,gamma_gamma_cdf(threshold,b,a),places=8)
            self.assertLess(abs(exact-np.mean(samples <= threshold)),.004)

    def test_stationary_mean_calibration(self):
        for pair in ('clear-haze','clear-rain'):
            previous_power = float('inf')
            for mean in (.014,.3,.5,.59,.7,.85):
                fso,rf,power = conditional_erasures(mean,pair)
                self.assertAlmostEqual(np.dot((2/3,1/3),fso),mean,places=8)
                self.assertGreater(fso[1],fso[0])
                self.assertTrue(all(0 <= p <= 1 for p in (*fso,*rf)))
                self.assertLess(power,previous_power)
                previous_power = power

    def test_markov_stationarity_transitions_and_correlation(self):
        state = markov_states(np.random.default_rng(9113),500000,.1,.2)
        self.assertLess(abs(state.mean()-1/3),.006)
        for start,target in ((0,.1),(1,.2)):
            mask = state[:-1] == start
            observed = np.mean(state[1:][mask] != start)
            self.assertLess(abs(observed-target),.003)
        self.assertLess(abs(np.corrcoef(state[:-1],state[1:])[0,1]-.7),.009)
        fso,_,_ = conditional_erasures(.59,'clear-haze')
        erased = np.random.default_rng(781).random(len(state)) < np.take(fso,state)
        predicted = (2/3)*(1/3)*(fso[1]-fso[0])**2*.7/(.59*(1-.59))
        self.assertLess(abs(np.corrcoef(erased[:-1],erased[1:])[0,1]-predicted),.009)

    def test_common_wall_clock_independent_of_window_and_method(self):
        cfg = Config()
        a = trial_inputs(cfg,1224)
        b = trial_inputs(replace(cfg,window=1),1224)
        self.assertEqual(a.fingerprint,b.fingerprint)
        self.assertEqual(a.state_at(0),a.state_at(a.epoch_ticks-1))
        for epoch in (1,3,17,41):
            tick = epoch*a.epoch_ticks
            self.assertEqual(a.state_at(tick),int(a.states[epoch]))
            self.assertEqual(a.state_at(tick),b.state_at(tick))
        with self.assertRaises(ValueError):
            a.states[0] = 1
        for method in METHODS:
            result = simulate(cfg,method,1224,bank=a)
            self.assertEqual(result['fingerprint'],a.fingerprint)

    def test_serializer_grid_and_persistent_channel_trace(self):
        for scale,pair in ((1.,'clear-haze'),(.1,'clear-haze'),(1.,'clear-rain')):
            cfg = Config(transition_scale=scale,weather_pair=pair)
            bank = trial_inputs(cfg,1714)
            r = simulate(cfg,'bitmap',1714,record_trace=True,bank=bank)
            self.assertTrue(r['success'])
            for event in r['events']:
                tick = event['time_s']*cfg.fso_rate_bps
                if event['event'] == 'data_launch':
                    self.assertAlmostEqual(tick,round(tick),places=6)
                    self.assertEqual(event['channel_state'],bank.state_at(round(tick)))
            m = metadata(cfg)
            self.assertAlmostEqual(m['stationary_fso_erasure'],cfg.eps_fso,places=8)


if __name__ == '__main__':
    unittest.main()
