"""Each architecture carries exactly one frozen (W, tau) into both six-curve figures."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from coding import Config
from experiments import sources,write_json,sha,variant_config
from feedback_options import ARCHITECTURES
from optimize_window import candidate_grid,select_global,read_optimization,ERASURES,KIND,REFERENCE,WINDOWS
from run_comparison import evaluation_cells, CURVES


class GlobalContractTests(unittest.TestCase):
    def selection(self, arch, cfg=None):
        grid = candidate_grid()
        training = {f'{e:g}':{k:{'t99':100.+i} for k in [*grid,REFERENCE]} for i,e in enumerate(ERASURES)}
        selected,_ = select_global(training,grid)
        cfg = cfg or replace(Config(),**ARCHITECTURES[arch])
        return {'manifest':{'kind':KIND,'architecture':arch,'source_hashes':sources(),'candidates':grid,
                'config':asdict(cfg),'windows':sorted(WINDOWS),'thresholds':[.25,.4,.5,.6],
                'erasure_points':list(ERASURES),'seed':100,'validation_seed':10000,'spacing':1000,
                'n_validation':100},'training':training,'selected_policy':selected,'validation':{}}

    def test_seven_cells_six_curves_and_frozen_policies(self):
        selections = {a:self.selection(a) for a in ARCHITECTURES}
        selections['hd_real']['selected_policy'] = {'window':64,'state_threshold':.25}
        cells = evaluation_cells(selections,50,200,40000)
        self.assertEqual(len(cells),7)
        for name,cfg,seed,n,variants in cells:
            self.assertEqual([v[0] for v in variants],list(CURVES))
            hd = variant_config(cfg,variants[2])
            self.assertEqual((hd.window,hd.state_threshold,hd.feedback_link,hd.feedback_ideal),
                             (64,.25,'half_duplex',False))
            ded = variant_config(cfg,variants[5])
            self.assertEqual((ded.feedback_link,ded.feedback_ideal),('dedicated',True))
            self.assertEqual(cfg.epoch_rf_packets,10)
        with self.assertRaises(ValueError): evaluation_cells(selections,50,200,101)
        selections['hd_ideal'] = self.selection(
            'hd_ideal',replace(Config(),epoch_rf_packets=1,**ARCHITECTURES['hd_ideal']))
        with self.assertRaises(ValueError): evaluation_cells(selections,50,200,40000)

    def test_variants_cannot_change_the_channel(self):
        with self.assertRaises(ValueError): variant_config(Config(),('x','bitmap',{'eps_fso':.3}))
        with self.assertRaises(ValueError): variant_config(Config(),('x','bitmap',{'epoch_rf_packets':1}))

    def test_read_rejects_modified_selection_and_wrong_architecture(self):
        selection = self.selection('ded_real')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'optimization.json'
            write_json(path.parent/'manifest.json',selection['manifest'])
            selection['manifest_sha256'] = sha(path.parent/'manifest.json')
            write_json(path,selection)
            self.assertEqual(read_optimization(path,'ded_real')['selected_policy'],selection['selected_policy'])
            with self.assertRaises(ValueError): read_optimization(path,'hd_real')
            selection['selected_policy']['window'] = 8
            write_json(path,selection)
            with self.assertRaises(ValueError): read_optimization(path)


if __name__ == '__main__':
    unittest.main()
