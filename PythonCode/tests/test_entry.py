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
    def test_padding_groups_select_canonical_names_and_preserve_shared_settings(self):
        groups = discover_experiments(ROOT / 'experiment_configs')
        self.assertNotIn('F5_Traditional_Content_HSV_Gradient', groups)
        expected = ['F1.1_Traditional_Content_HSV48', 'F2.1_Traditional_Content_HSV_Gradient192']
        spec = importlib.util.spec_from_file_location('entry', ROOT / '02_SOP_Experiments.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        self.assertEqual(entry.select_groups(groups, 'F1.1,F2.1'), expected)
        with self.assertRaises(ValueError):
            entry.select_groups(groups, 'F1.2')
        self.assertEqual(entry.select_groups(groups, 'F2.1,F2.1'), expected[1:])
        with self.assertRaises(ValueError):
            entry.select_groups(groups, 'F5')
        configs = [load_config(ROOT / 'config.json', experiment_path=groups[g][0]) for g in expected]
        self.assertEqual(list(configs[0]['feature_sets']), ['ContentHSV48'])
        self.assertEqual(list(configs[1]['feature_sets']), ['ContentHSV_HOG192'])
        self.assertEqual(configs[0]['feature_sets']['ContentHSV48'], {'extractor': 'hsv_hog_content', 'columns': [0, 48]})
        self.assertEqual(configs[1]['feature_sets']['ContentHSV_HOG192'], {'extractor': 'hsv_hog_content', 'columns': [0, 192]})
        for config in configs:
            self.assertEqual(config['run_name'], 'Main_Experiments')
            self.assertEqual((config['outer_folds'], config['inner_folds']), (10, 3))
            self.assertEqual(config['target_validation_recall'], .95)
            self.assertEqual(config['comparisons'], [])
        excluded = {'feature_profile', 'feature_sets'}
        self.assertEqual({k: v for k, v in configs[0].items() if k not in excluded},
                         {k: v for k, v in configs[1].items() if k not in excluded})

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
        self.assertIn('All requested runs finished.', result.stdout)

    def test_exit_watchdog_records_hang_and_cleans_up_normal_exit(self):
        def run(script):
            return subprocess.run([sys.executable, '-c', textwrap.dedent(script)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=60)
        with tempfile.TemporaryDirectory() as folder:
            common = f'''
                import importlib.util, threading, time
                spec = importlib.util.spec_from_file_location('entry', {str(ROOT / '02_SOP_Experiments.py')!r})
                entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
            '''
            normal = run(common + f'''
                entry.arm_exit_watchdog({folder!r}, seconds=30)
            ''')
            self.assertEqual(normal.returncode, 0, normal.stderr)
            self.assertFalse((Path(folder) / 'exit_watchdog_traceback.txt').exists())
            # A non-daemon thread keeps the interpreter from exiting beyond the timeout.
            hang = run(common + f'''
                entry.arm_exit_watchdog({folder!r}, seconds=1)
                threading.Thread(target=time.sleep, args=(3,), name='blocking-worker').start()
            ''')
            self.assertEqual(hang.returncode, 0, hang.stderr)
            text = (Path(folder) / 'exit_watchdog_traceback.txt').read_text(encoding='utf-8')
            self.assertIn('File "', text)

    def test_handcrafted_baselines_do_not_require_torch(self):
        spec = importlib.util.spec_from_file_location('entry', ROOT / '02_SOP_Experiments.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        config = load_config(ROOT / 'config.json',
                             experiment_path=ROOT / 'experiment_configs' / 'F1_Traditional_HSV48.json')
        with patch.object(entry.importlib, 'import_module', side_effect=AssertionError('Unexpected torch import')):
            entry.prepare_runtime([config], 'baselines')

    def test_resnet50_group_initializes_torch_before_baselines(self):
        spec = importlib.util.spec_from_file_location('entry', ROOT / '02_SOP_Experiments.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        config = load_config(ROOT / 'config.json',
                             experiment_path=ROOT / 'experiment_configs' / 'F6_ResNet50_Global2048.json')
        self.assertEqual(config['feature_sets']['ResNet50_Global2048']['extractor'], 'resnet50')
        import contextlib
        import io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            entry.prepare_runtime([config], 'baselines')
        # A ResNet50 group must load torch before baselines even without SupCon.
        self.assertIn('PyTorch ready:', output.getvalue())

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

    def test_region_max_transform_plan_a(self):
        # Plan A (F4.1): global columns kept, four region vectors reduced by per-channel max.
        values = np.zeros((2, 12), dtype=np.float32)
        values[:, :4] = [1, 2, 3, 4]                         # global (4 channels here)
        values[0, 4:] = [0.9, 0.1, 0.2, 0.1, 0.1, 0.8, 0.3, 0.0]  # two regions x 4 channels
        values[1, 4:] = [0.1, 0.8, 0.3, 0.0, 0.9, 0.1, 0.2, 0.1]  # same regions, swapped order
        config = {'conditions': {'clean': {}}, 'feature_sets': {'A': {'components': [
            {'extractor': 'hsv_hog', 'columns': [0, 4]},
            {'extractor': 'hsv_hog', 'columns': [4, 12], 'transform': 'region_max', 'regions': 2}]}}}
        with patch('utils.sop_feature_bank.extract_cached', return_value=values):
            bank = build_banks(None, None, config, 'test', None)['A']['clean']
        np.testing.assert_allclose(bank, [[1, 2, 3, 4, .9, .8, .3, .1]] * 2, rtol=0, atol=1e-7)

    def test_plan_a_group_and_dotted_names(self):
        from utils.sop_config import validate_config
        groups = discover_experiments(ROOT / 'experiment_configs')
        names = list(groups)
        self.assertLess(names.index('F4_ResNet_GlobalLocal2560'), names.index('F4.1_ResNet_GlobalRegionMax1024'))
        self.assertLess(names.index('F4.1_ResNet_GlobalRegionMax1024'), names.index('F4.2_ResNet_GlobalCropMax1024'))
        spec = importlib.util.spec_from_file_location('entry', ROOT / '02_SOP_Experiments.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        self.assertEqual(entry.select_groups(groups, 'F4.1'), ['F4.1_ResNet_GlobalRegionMax1024'])
        self.assertEqual(entry.select_groups(groups, 'F4'), ['F4_ResNet_GlobalLocal2560'])
        config = load_config(ROOT / 'config.json', experiment_path=groups['F4.1_ResNet_GlobalRegionMax1024'][0])
        self.assertEqual(config['feature_profile'], 'F4.1_ResNet_GlobalRegionMax1024')
        for field, bad in (('feature_profile', '..'), ('feature_profile', '.hidden'), ('run_name', 'run.1')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_config(dict(config, **{field: bad}))
        broken = json.loads(json.dumps(config))
        broken['feature_sets']['ResNet18_GlobalRegionMax1024']['components'][1]['columns'] = [512, 2559]
        with self.assertRaisesRegex(ValueError, 'split evenly'):
            validate_config(broken)

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
