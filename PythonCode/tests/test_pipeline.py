"""Synthetic execution checks; not formal SOP experimental evidence."""
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sop_config import load_config
from sop_experiment import Experiment
from sop_protocol import write_json
from sop_supcon import SupConLoss, product_batches, fit_representation, save_model, load_model, matching_scores
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / '.verification_artifacts'
ARTIFACTS.mkdir(exist_ok=True)

class PipelineTests(unittest.TestCase):
    def test_preprocessing_zip_audit(self):
        from sop_preprocessing import main
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            root = Path(folder)
            archive = root / 'input.zip'
            with ZipFile(archive, 'w') as zf:
                train = ['image_id class_id super_class_id path']
                test = ['image_id class_id super_class_id path']
                for image_id, sku, split, size, missing in [
                    (1, 1, 'train', 120, False), (2, 1, 'train', 120, False),
                    (3, 2, 'train', 120, False), (4, 2, 'train', 80, False),
                    (5, 3, 'train', 120, True), (6, 4, 'test', 120, False),
                    (7, 4, 'test', 120, False)]:
                    name = f'images/{image_id}.jpg'
                    (train if split == 'train' else test).append(f'{image_id} {sku} 1 {name}')
                    if not missing:
                        output = io.BytesIO()
                        Image.new('RGB', (size, size), (120, 50, 80)).save(output, format='JPEG')
                        zf.writestr('SOP/' + name, output.getvalue())
                zf.writestr('SOP/Ebay_train.txt', '\n'.join(train))
                zf.writestr('SOP/Ebay_test.txt', '\n'.join(test))
            with patch.object(sys, 'argv', ['01', '--zip-path', str(archive), '--output-dir', str(root / 'data')]):
                main()
            frame = pd.read_csv(root / 'data' / 'clean_manifest.csv')
            self.assertEqual(frame.image_id.tolist(), [1, 2, 6, 7])
            reasons = pd.read_csv(root / 'data' / 'excluded_samples.csv').reason.tolist()
            self.assertEqual(set(reasons), {'only_one_eligible_image_for_product', 'original_smaller_than_100', 'missing_file_in_zip'})
            for relative in frame.processed_relative_path:
                with Image.open(root / 'data' / relative) as image:
                    self.assertEqual(image.size, (224, 224))

    def test_loss_equation_and_gradient(self):
        x = np.random.default_rng(12).normal(size=(6, 7))
        labels = np.repeat(np.arange(3), 2)
        z = x / np.linalg.norm(x, axis=1, keepdims=True)
        logits = z @ z.T / .1
        expected = []
        for i in range(6):
            others = np.arange(6) != i
            positive = (labels == labels[i]) & others
            maximum = logits[i, others].max()
            denominator = maximum + np.log(np.exp(logits[i, others] - maximum).sum())
            expected.append(-(logits[i, positive] - denominator).mean())
        value = torch.tensor(x, requires_grad=True)
        loss = SupConLoss(.1)(value, torch.tensor(labels))
        self.assertAlmostEqual(loss.item(), np.mean(expected), places=10)
        loss.backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue(torch.autograd.gradcheck(lambda v: SupConLoss(.1)(v, torch.tensor(labels)), (value,)))
        self.assertAlmostEqual(SupConLoss(.1)(torch.tensor(x.reshape(3, 2, 7)), torch.arange(3)).item(), loss.item(), places=10)
        with self.assertRaises(ValueError):
            SupConLoss()(torch.randn(3, 4), torch.arange(3))

    def test_tail_scaling_and_reload_512_dimensions(self):
        frame = pd.DataFrame({'class_id': np.repeat(np.arange(6), 2)})
        settings = copy.deepcopy(load_config(ROOT / 'experiment_config.json', smoke=True)['supcon'])
        settings.update(products_per_batch=2, hidden_dim=12, representation_dim=8, projection_dim=8, device='cpu')
        batches = list(product_batches(frame, list(range(5)), settings, np.random.default_rng(1)))
        self.assertEqual(sorted(np.concatenate([x for x, _ in batches]).tolist()), list(range(10)))
        self.assertTrue(all(len(np.unique(y)) >= 2 for _, y in batches))
        features = np.random.default_rng(2).normal(size=(12, 512)).astype(np.float32)
        features[10:] += 1000
        fitted = fit_representation(frame, features, list(range(5)), .1, settings, 1)
        np.testing.assert_allclose(fitted['mean'], features[:10].mean(axis=0), atol=1e-6)
        pairs = pd.DataFrame({'query_row': [0, 2, 10], 'reference_row': [1, 3, 11]})
        bank = {'clean': features}
        before = matching_scores(fitted, pairs, bank)
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            path = Path(folder) / 'model.pt'
            save_model(fitted, path, {'test': True})
            np.testing.assert_array_equal(before, matching_scores(load_model(path), pairs, bank))
        with torch.no_grad():
            for p in fitted['model'].projector.parameters():
                p.fill_(100)
        np.testing.assert_array_equal(before, matching_scores(fitted, pairs, bank))
        from sop_baselines import fit_classifier, scores
        for algorithm in ('LR', 'SVM'):
            model = fit_classifier(features[:10], np.tile([0, 1], 5), algorithm, 1, 1)
            self.assertTrue(np.isfinite(scores(model, features[10:])).all())

    def test_resnet_geometry(self):
        import torchvision.models as models
        from sop_feature_bank import ResNet18Extractor
        torch.set_num_threads(2)
        with patch('torchvision.models.resnet18', return_value=models.resnet18(weights=None)):
            extractor = ResNet18Extractor({'device': 'cpu'})
        images = [Image.new('RGB', (224, 224), (50, 100, 150))]
        values = extractor(images)
        self.assertEqual(values.shape, (1, 2560))
        self.assertTrue(np.isfinite(values).all())
        self.assertFalse(any(p.requires_grad for p in extractor.model.parameters()))
        np.testing.assert_array_equal(values, extractor(images))

    def test_shared_protocol_end_to_end_resume(self):
        from sop_baselines import run_experiment as baselines
        from sop_supcon import run_experiment as supcon
        from sop_reporting import combine_results
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            root = Path(folder)
            rows = []
            for category in range(3):
                for product in range(8):
                    sku = category * 8 + product
                    for view in range(4):
                        image_id = len(rows) + 1
                        relative = f'images/train/{sku}/{image_id}.jpg'
                        path = root / 'data' / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        arr = np.clip(np.random.default_rng(sku).integers(20, 230, (224, 224, 3)) + view, 0, 255).astype(np.uint8)
                        Image.fromarray(arr).save(path)
                        rows.append(dict(image_id=image_id, class_id=sku, super_class_id=category, source_split='train', original_width=224, original_height=224, processed_relative_path=relative))
            pd.DataFrame(rows).to_csv(root / 'data' / 'clean_manifest.csv', index=False)
            write_json(root / 'data' / 'audit_summary.json', {'clean_images': len(rows), 'complete': True})
            config = load_config(ROOT / 'experiment_config.json', smoke=True)
            config.update(minimum_train_images=96, run_name='synthetic', train_anchor_cap=8, evaluation_anchor_cap=6)
            for key, name in [('data_dir', 'data'), ('protocol_dir', 'protocol'), ('cache_dir', 'cache'), ('results_dir', 'results')]:
                config[key] = str(root / name)
            config['supcon'].update(hidden_dim=16, representation_dim=8, projection_dim=8, device='cpu')
            experiment = Experiment(config)
            for fold in (1, 2):
                context = experiment.context(fold)
                self.assertFalse(set(context['train_products']) & set(context['test_products']))
                for train, validation, _, _ in context['inner']:
                    self.assertFalse(set(train) & set(validation))
            snapshots = {p.relative_to(root / 'protocol'): p.read_bytes() for p in (root / 'protocol').rglob('*pairs.csv')}
            first = baselines(experiment)
            second = supcon(Experiment(config))
            combine_results(experiment)
            self.assertTrue((second.parent / 'combined_summary_mean_sd.csv').exists())
            for rel, value in snapshots.items():
                self.assertEqual(value, (root / 'protocol' / rel).read_bytes())
            self.assertEqual(baselines(Experiment(config)), first)
            self.assertEqual(supcon(Experiment(config)), second)
            self.assertTrue(pd.read_csv(second / 'summary_mean_sd.csv').outer_fold_count.eq(2).all())
            other = copy.deepcopy(config)
            other['feature_profile'] = 'alternative'
            other['feature_sets'] = {'Alternative64': {'extractor': 'hsv_hog', 'columns': [0, 64]}}
            other['comparisons'] = []
            changed = Experiment(other)
            self.assertEqual(changed.protocol_id, experiment.protocol_id)
            pd.testing.assert_frame_equal(changed.context(1)['test_pairs'], experiment.context(1)['test_pairs'])
            self.assertEqual(changed.banks['Alternative64']['clean'].shape[1], 64)
            from sop_reporting import report_results
            synthetic_rows = pd.read_csv(first / 'outer_fold_metrics.csv')
            synthetic_rows = synthetic_rows[synthetic_rows.features.eq('HSV48')].copy()
            synthetic_rows['features'] = 'Alternative64'
            report_results(synthetic_rows, [], root / 'single_feature_report', other)
            self.assertTrue((root / 'single_feature_report' / 'paired_delta_mean_sd.csv').exists())
            path = root / 'protocol' / 'outer_01' / 'train_pairs.csv'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                experiment.context(1)

if __name__ == '__main__':
    unittest.main()
