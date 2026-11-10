"""Global (W, tau) selection, finite objective and paired statistics."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from optimize_window import candidate_grid, select_global, ERASURES, REFERENCE
from statistics_tools import quantile, quantile_interval, paired_gain, paired_outage_reduction


class OptimizationTests(unittest.TestCase):
    def training(self, candidates):
        return {f'{e:g}': {k:{'t99':100.*(i+1)} for k in [*candidates, REFERENCE]}
                for i,e in enumerate(ERASURES)}

    def test_small_sample_tail_interval_does_not_fake_finite_upper_endpoint(self):
        self.assertTrue(np.isinf(quantile_interval(np.arange(1,101,dtype=float))[1]))
        self.assertTrue(np.isfinite(quantile_interval(np.arange(1,2001,dtype=float))[1]))

    def test_grid_and_disabled_state_rule_is_ablation_only(self):
        grid = candidate_grid()
        self.assertEqual(len(grid),36)
        self.assertTrue(all(p['state_threshold'] > 0 for p in grid.values()))
        for kwargs in ({'windows':[1,12]}, {'thresholds':[0.,.5]}, {'thresholds':[]}):
            with self.assertRaises(ValueError):
                candidate_grid(**kwargs)

    def test_pooling_normalization_and_declared_tie(self):
        grid = candidate_grid()
        training = self.training(grid)
        selected, scores = select_global(training,grid)
        self.assertEqual(selected,grid['w512_tau0.25'])
        self.assertTrue(all(s['objective']==1 for s in scores.values()))
        # A global compromise, not a table of independently optimal policies.
        for i,e in enumerate(ERASURES):
            training[f'{e:g}']['w16_tau0.5']['t99'] *= .95
            training[f'{e:g}']['w12_tau0.4']['t99'] *= .98 if i<5 else .9
        selected, scores = select_global(training,grid)
        self.assertEqual(selected,grid['w16_tau0.5'])
        self.assertAlmostEqual(scores['w16_tau0.5']['objective'],.95)
        self.assertLess(training['0.85']['w12_tau0.4']['t99'],training['0.85']['w16_tau0.5']['t99'])

    def test_bad_reference_missing_scenario_and_failures_are_not_discarded(self):
        grid = candidate_grid()
        training = self.training(grid)
        training['0.85']['w8_tau0.5']['t99'] = float('inf')
        _,scores = select_global(training,grid)
        self.assertTrue(np.isinf(scores['w8_tau0.5']['objective']))
        training['0.85'][REFERENCE]['t99'] = None
        with self.assertRaises(ValueError):
            select_global(training,grid)
        del training['0.85']
        with self.assertRaises(ValueError):
            select_global(training,grid)
        times = np.r_[np.ones(98),np.inf,np.inf]
        self.assertTrue(np.isinf(quantile(times)))
        self.assertEqual(paired_gain(times,times,20),(None,None))

    def test_paired_gain_sign_reproducibility_and_outage_reduction(self):
        base = np.arange(1,101,dtype=float)
        candidate = base*.9
        a = paired_gain(base,candidate,100,913)
        self.assertEqual(a,paired_gain(base,candidate,100,913))
        self.assertAlmostEqual(a[0],10)
        self.assertGreater(a[1][0],0)
        gain,ci = paired_outage_reduction(base,candidate,95)
        self.assertAlmostEqual(gain,5)
        self.assertLess(ci[0],gain)
        self.assertGreater(ci[1],gain)
        gain,ci = paired_outage_reduction(base,base,95)
        self.assertEqual(gain,0)
        self.assertLess(ci[0],0)
        self.assertGreater(ci[1],0)


if __name__ == '__main__':
    unittest.main()
