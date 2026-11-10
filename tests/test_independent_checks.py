"""Independent GF(2), feedback-accounting and input/output integrity checks."""
from dataclasses import replace
import ast
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from coding import Config, Codebook, Decoder
from channel import trial_inputs
from experiments import COLUMNS, compatible_manifest, load_samples
from model import simulate


def add_basis(basis,row):
    while row:
        pivot = row.bit_length()-1
        if pivot not in basis:
            basis[pivot] = row
            return
        row ^= basis[pivot]


def in_span(basis,row):
    while row:
        pivot = row.bit_length()-1
        if pivot not in basis:
            return False
        row ^= basis[pivot]
    return True


class IndependentChecks(unittest.TestCase):
    def test_all_matplotlib_axis_math_labels_render(self):
        from matplotlib.mathtext import MathTextParser
        parser = MathTextParser('agg')
        tree = ast.parse((ROOT/'python/plot_results.py').read_text())
        labels = []
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                for keyword in node.keywords:
                    if keyword.arg in ('xlabel','ylabel') and isinstance(keyword.value,ast.Constant):
                        label = keyword.value.value
                        if isinstance(label,str) and '$' in label:
                            labels.append(label)
                            parser.parse(label,dpi=72)
        self.assertEqual(len(labels),4)

    def test_all_peeling_recoveries_identifiable_by_independent_gf2_elimination(self):
        cfg = Config(k=36,r=4).public()
        book = Codebook(cfg,2744)
        decoder = Decoder(cfg,book.precode)
        basis = {}
        for i,neighbors in enumerate(book.precode):
            add_basis(basis,(1 << (cfg.k+i)) ^ sum(1 << int(j) for j in neighbors))
        ids = np.random.default_rng(3902).permutation(2*cfg.attempt_cap+cfg.k_hat)[:160]
        for graph_id in ids:
            neighbors = book.neighbors(int(graph_id))
            add_basis(basis,sum(1 << int(j) for j in neighbors))
            decoder.add(neighbors)
            for symbol in np.flatnonzero(decoder.known):
                self.assertTrue(in_span(basis,1 << int(symbol)))
            if decoder.done():
                self.assertTrue(all(in_span(basis,1 << j) for j in range(cfg.k)))
        self.assertTrue(decoder.done())

    def test_feedback_and_timing_for_extreme_grid_windows_and_long_epochs(self):
        cfg = Config(epoch_rf_packets=100)
        bank = trial_inputs(cfg,2747)
        for window in (1,2,64):
            r = simulate(replace(cfg,window=window),'bitmap',2747,bank=bank)
            self.assertTrue(r['success'])
            self.assertEqual(r['fingerprint'],bank.fingerprint)
            self.assertEqual(r['bitmap_payload_bits_launched'],window*r['bitmap_reports_launched'])
            self.assertLessEqual(r['data_airtime_to_observation_s'][1],r['observation_end_s']+1e-12)
            self.assertLessEqual(r['bitmap_airtime_to_observation_s'],r['observation_end_s']+1e-12)
            self.assertEqual(sum(r['completed']),sum(r['received'])+sum(r['erased']))

    def test_manifest_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manifest.json'
            compatible_manifest(path,{'window':12,'seeds':[1,2]})
            compatible_manifest(path,{'window':12,'seeds':[1,2]})
            with self.assertRaises(ValueError):
                compatible_manifest(path,{'window':16,'seeds':[1,2]})

    def test_sample_loader_rejects_wrong_seeds_and_packet_accounting(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            cell = out/'samples'/'toy'
            cell.mkdir(parents=True)
            rows = np.zeros((2,len(COLUMNS)))
            rows[:,0] = [40,41]
            rows[:,1:3] = [1,100]
            rows[:,4:9] = [10,9,6,3,1]
            rows[:,9:12] = [12,1,1]
            rows[:,21:23] = [25,1]
            file = cell/'sample.npz'
            np.savez_compressed(file,bitmap=rows)
            load_samples(out,'toy',40,2,['bitmap'],{'bitmap':12})
            with self.assertRaises(ValueError):
                load_samples(out,'toy',39,2,['bitmap'],{'bitmap':12})
            rows[0,7] = 4
            np.savez_compressed(file,bitmap=rows)
            with self.assertRaises(ValueError):
                load_samples(out,'toy',40,2,['bitmap'],{'bitmap':12})


if __name__ == '__main__':
    unittest.main()
