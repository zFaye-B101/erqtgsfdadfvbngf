"""Checks for dynamic groups, feature composition and saved comparisons."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.sop_config import discover_experiments, load_config
from utils.sop_feature_bank import build_banks
from utils.sop_reporting import update_comparisons


class EntryTests(unittest.TestCase):
    def test_all_initializes_torch_before_experiments_and_reports(self):
        # Use a fresh process: the pipeline tests already import torch globally.
        script = textwrap.dedent('''
            import importlib.util
            import sys
            import types
            from pathlib import Path
            from unittest.mock import patch

            spec = importlib.util.spec_from_file_location('entry', '02_SOP_Experiments.py')
            entry = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(entry)
            assert 'torch' not in sys.modules
            events = []

            def experiment(config):
                assert 'torch' in sys.modules, 'PyTorch loaded after experiment setup'
                assert 'torchvision' in sys.modules, 'ResNet dependencies were not checked'
                events.append('setup')
                return object()

            def baseline_run(experiment):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, axis = plt.subplots()
                axis.plot([0, 1], [0, 1])
                fig.canvas.draw()
                plt.close(fig)
                events.append('baseline')
                return Path('unused')

            def supcon_run(experiment):
                import torch
                assert torch.ones(1).item() == 1.0
                events.append('supcon')
                return Path('unused')

            baselines = types.ModuleType('utils.sop_baselines')
            baselines.run_experiment = baseline_run
            supcon = types.ModuleType('utils.sop_supcon')
            supcon.run_experiment = supcon_run
            with patch.dict(sys.modules, {'utils.sop_baselines': baselines, 'utils.sop_supcon': supcon}), \\
                 patch('utils.sop_experiment.Experiment', side_effect=experiment), \\
                 patch('utils.sop_reporting.print_tables'), \\
                 patch('utils.sop_reporting.combine_results'), \\
                 patch.object(entry, 'update_comparisons'):
                entry.main(['--group', 'F1,F2,F3,F4', '--method', 'all'])
            assert events == ['setup', 'baseline', 'supcon'] * 4, events
        ''')
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PyTorch ready:', result.stdout)

    def test_handcrafted_baselines_do_not_require_torch(self):
        spec = importlib.util.spec_from_file_location('entry', ROOT / '02_SOP_Experiments.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        config = load_config(ROOT / 'config.json',
                             experiment_path=ROOT / 'experiment_configs' / 'F1_Traditional_HSV48.json')
        with patch.object(entry.importlib, 'import_module', side_effect=AssertionError('Unexpected torch import')):
            entry.prepare_runtime([config], 'baselines')

    def test_dynamic_group_and_common_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'F8.json'
            path.write_text(json.dumps({'group': 'F8_New', 'features': {
                'HSV': {'extractor': 'hsv_hog', 'columns': [0, 48]}}}))
            groups = discover_experiments(folder)
            self.assertEqual(list(groups), ['F8_New'])
            config = load_config(ROOT / 'config.json', experiment_path=path)
            self.assertEqual(config['feature_profile'], 'F8_New')
            self.assertEqual(Path(config['data_dir']), ROOT / 'processed_sop')
            spec = importlib.util.spec_from_file_location('entry', ROOT / '02_SOP_Experiments.py')
            entry = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(entry)
            self.assertEqual(entry.select_groups(groups, 'F8'), ['F8_New'])

    def test_component_concatenation(self):
        config = {'conditions': {'clean': {}}, 'feature_sets': {'Combo': {'components': [
            {'extractor': 'hsv_hog', 'columns': [0, 2], 'normalization': 'l2'},
            {'extractor': 'hsv_hog', 'columns': [2, 3]}]}}}
        values = np.zeros((2, 192), dtype=np.float32)
        values[:, :3] = [3, 4, 7]
        with patch('utils.sop_feature_bank.extract_cached', return_value=values) as extract:
            bank = build_banks(None, None, config, 'test', None)
        np.testing.assert_allclose(bank['Combo']['clean'], [[.6, .8, 7], [.6, .8, 7]])
        self.assertEqual(extract.call_count, 1)

    def test_saved_comparison_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, value, protocol in [('F1', .2, 'same'), ('F2', .1, 'same'), ('F8', .3, 'other')]:
                directory = root / name / 'baselines'
                directory.mkdir(parents=True)
                identity = {'shared_protocol_id': protocol, 'code_fingerprint': 'same',
                            'config': {'outer_folds': 2, 'feature_profile': name, 'feature_sets': {name: {}}, 'comparisons': []}}
                (directory / 'protocol.json').write_text(json.dumps(identity))
                rows = [{'algorithm': 'LR', 'features': name, 'condition': 'clean', 'negative_subset': 'all',
                         'fold': fold, 'FPR': value, 'TPR': .95, 'F1': .8, 'ROC_AUC': .9} for fold in [1, 2]]
                pd.DataFrame(rows).to_csv(directory / 'outer_fold_metrics.csv', index=False)
            update_comparisons(root)
            deltas = pd.read_csv(root / 'comparisons' / 'paired_group_deltas.csv')
            self.assertEqual(set(deltas.added_group), {'F2'})
            np.testing.assert_allclose(deltas.loc[deltas.metric == 'FPR', 'delta_added_minus_baseline'], -.1)
            skipped = json.loads((root / 'comparisons' / 'skipped_incompatible_comparisons.json').read_text())
            self.assertEqual(len(skipped), 2)

    def test_empty_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            update_comparisons(folder)
            self.assertTrue((Path(folder) / 'comparisons' / 'paired_group_deltas.csv').exists())


if __name__ == '__main__':
    unittest.main()
