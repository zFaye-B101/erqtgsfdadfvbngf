"""Synthetic execution checks; not formal SOP experimental evidence."""
import copy
import io
from pathlib import Path
import sys
import tempfile
import runpy
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from utils.sop_config import load_config
from utils.sop_experiment import Experiment
from utils.sop_protocol import write_json
from utils.sop_supcon import SupConLoss, product_batches, fit_representation, save_model, load_model, matching_scores
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / '.verification_artifacts'
ARTIFACTS.mkdir(exist_ok=True)

class PipelineTests(unittest.TestCase):
    def test_preprocessing_zip_audit(self):
        main = runpy.run_path(str(ROOT / "01_SOP_Preprocessing.py"))["main"]
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
        settings = copy.deepcopy(load_config(ROOT / 'config.json', smoke=True)['supcon'])
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
        from utils.sop_baselines import fit_classifier, scores
        for algorithm in ('LR', 'SVM'):
            model = fit_classifier(features[:10], np.tile([0, 1], 5), algorithm, 1, 1)
            self.assertTrue(np.isfinite(scores(model, features[10:])).all())

    def test_resnet_geometry(self):
        import torchvision.models as models
        from utils.sop_feature_bank import ResNet18Extractor
        torch.set_num_threads(2)
        with patch('torchvision.models.resnet18', return_value=models.resnet18(weights=None)):
            extractor = ResNet18Extractor({'device': 'cpu'})
        images = [Image.new('RGB', (224, 224), (50, 100, 150))]
        values = extractor(images)
        self.assertEqual(values.shape, (1, 2560))
        self.assertTrue(np.isfinite(values).all())
        self.assertFalse(any(p.requires_grad for p in extractor.model.parameters()))
        np.testing.assert_array_equal(values, extractor(images))

    def test_resnet50_global_and_regions(self):
        import torchvision.models as models
        from utils.sop_feature_bank import EXTRACTORS, ResNet50Extractor, ResNet50RegionsExtractor
        torch.set_num_threads(2)
        backbone = models.resnet50(weights=None)  # random weights: no download, shape check only
        with patch('torchvision.models.resnet50', return_value=backbone) as factory:
            global_extractor = ResNet50Extractor({'device': 'cpu'})
            region_extractor = ResNet50RegionsExtractor({'device': 'cpu'})
        self.assertEqual(str(factory.call_args.kwargs['weights']), 'ResNet50_Weights.IMAGENET1K_V2')
        self.assertIs(EXTRACTORS['resnet50'], ResNet50Extractor)
        images = [Image.new('RGB', (224, 224), (50, 100, 150)), Image.new('RGB', (224, 224), (200, 30, 30))]
        global_values = global_extractor(images)
        region_values = region_extractor(images)
        self.assertEqual(global_values.shape, (2, 2048))
        self.assertEqual(region_values.shape, (2, 10240))
        self.assertTrue(np.isfinite(region_values).all())
        np.testing.assert_allclose(region_values[:, :2048], global_values, rtol=1e-5, atol=1e-6)
        self.assertFalse(any(p.requires_grad for p in global_extractor.model.parameters()))
        np.testing.assert_array_equal(global_values, global_extractor(images))
        self.assertEqual(global_extractor.metadata['weights'], 'IMAGENET1K_V2')

    def test_shared_protocol_end_to_end_resume(self):
        from utils.sop_baselines import run_experiment as baselines
        from utils.sop_supcon import run_experiment as supcon
        from utils.sop_reporting import combine_results
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            root = Path(folder)
            rows = []
            for category in range(3):
                for product in range(16):
                    sku = category * 16 + product
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
            config = load_config(ROOT / 'config.json', smoke=True)
            config.update(minimum_train_images=96, run_name='synthetic', train_anchor_cap=8, evaluation_anchor_cap=6)
            for key, name in [('data_dir', 'data'), ('protocol_dir', 'protocol'), ('cache_dir', 'cache'), ('results_dir', 'results')]:
                config[key] = str(root / name)
            config['supcon'].update(hidden_dim=16, representation_dim=8, projection_dim=8, device='cpu')
            experiment = Experiment(config)
            for fold in (1, 2):
                context = experiment.context(fold)
                self.assertFalse(set(context['train_products']) & set(context['test_products']))
                self.assertFalse(set(context['calibration_products']) & set(context['train_products']))
                self.assertFalse(set(context['calibration_products']) & set(context['test_products']))
                for train, validation, _, _ in context['inner']:
                    self.assertFalse(set(train) & set(validation))
                    self.assertFalse(set(context['calibration_products']) & (set(train) | set(validation)))
            snapshots = {p.relative_to(root / 'protocol'): p.read_bytes() for p in (root / 'protocol').rglob('*pairs.csv')}
            first = baselines(experiment)
            second = supcon(Experiment(config))
            combine_results(experiment)
            self.assertTrue((second.parent / 'combined_summary_mean_sd.csv').exists())
            for directory in (first, second):
                records = list(directory.rglob('calibration.json'))
                self.assertEqual(len(records), 8 if directory == first else 4)
                for path in records:
                    import json
                    record = json.loads(path.read_text())
                    self.assertFalse(set(record['calibration_products']) & set(record['fitting_products']))
                    predictions = pd.read_csv(path.with_name('calibration_predictions.csv.gz'))
                    if directory == second:
                        fitted = load_model(path.with_name('model.pt'))
                        self.assertEqual(fitted['training_products'], record['fitting_products'])
                        name = 'HSV_HOG192' if path.parent.name.endswith('HSV_HOG192') else 'HSV48'
                        actual = matching_scores(fitted, predictions, experiment.banks[name])
                    else:
                        import pickle
                        from utils.sop_baselines import matrix, scores
                        with path.with_name('model.pkl').open('rb') as stream:
                            fitted = pickle.load(stream)
                        self.assertEqual(fitted['fitting_products'], record['fitting_products'])
                        actual = scores(fitted, matrix(predictions, experiment.banks[fitted['features']]))
                    np.testing.assert_allclose(actual, predictions.score.to_numpy(), atol=1e-7)
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
            content = copy.deepcopy(config)
            content['feature_profile'] = 'content'
            content['feature_sets'] = {'ContentHSV48': {'extractor': 'hsv_hog_content', 'columns': [0, 48]}}
            content['comparisons'] = []
            content_experiment = Experiment(content)
            self.assertEqual(content_experiment.protocol_id, experiment.protocol_id)
            pd.testing.assert_frame_equal(content_experiment.context(1)['test_pairs'], experiment.context(1)['test_pairs'])
            from utils.sop_reporting import report_results
            synthetic_rows = pd.read_csv(first / 'outer_fold_metrics.csv')
            synthetic_rows = synthetic_rows[synthetic_rows.features.eq('HSV48')].copy()
            synthetic_rows['features'] = 'Alternative64'
            report_results(synthetic_rows, [], root / 'single_feature_report', other)
            self.assertTrue((root / 'single_feature_report' / 'paired_delta_mean_sd.csv').exists())
            path = root / 'protocol' / 'outer_01' / 'train_pairs.csv'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                experiment.context(1)

    def test_content_padding_and_temperature_expansion(self):
        from types import SimpleNamespace
        from utils.sop_features import resize_for_model, image_feature_vector
        from utils.sop_protocol import content_box
        from utils.sop_tuning import search_temperatures
        images, content = [], []
        for size in [(300, 300), (600, 150)]:
            image = resize_for_model(Image.new('RGB', size, (200, 30, 30)))
            row = SimpleNamespace(original_width=size[0], original_height=size[1])
            images.append(image_feature_vector(image))
            content.append(image_feature_vector(image.crop(content_box(row, 224))))
        self.assertGreater(np.abs(images[0][:48] - images[1][:48]).sum(), .1)
        np.testing.assert_array_equal(content[0], content[1])
        seen = []
        def evaluate(value):
            seen.append(value)
            return {'mean_auc': 1 - abs(value - .4)}
        best, grid, records, history = search_temperatures(evaluate, [.05, .1, .2], 2, 2, [.005, 1])
        self.assertEqual(best, .4)
        self.assertEqual(grid, [.2, .4, .8])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(history), 3)
        best, grid, records, history = search_temperatures(lambda value: {'mean_auc': value}, [.25, .5, 1], 2, 2, [.005, 1])
        self.assertEqual(best, 1)
        self.assertEqual(len(history), 1)

    def test_pair_features_and_official_query_coverage(self):
        from utils.sop_protocol import pair_matrix
        pairs = pd.DataFrame({'query_row': [0], 'reference_row': [1]})
        bank = {'clean': np.array([[1., 2., 3.], [3., 5., 7.]])}
        np.testing.assert_array_equal(pair_matrix(pairs, bank, [0, 3]), [[2., 3., 4.]])
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            experiment = object.__new__(Experiment)
            experiment.config = load_config(ROOT / 'config.json', smoke=True)
            experiment.config['run_official_test'] = True
            records = []
            for category in range(2):
                for product in range(16):
                    for view in range(2):
                        records.append({'class_id': category * 16 + product, 'super_class_id': category,
                                        'image_id': len(records), 'source_split': 'train' if product < 12 else 'test'})
            experiment.frame = pd.DataFrame(records)
            experiment.selected = sorted(experiment.frame.loc[experiment.frame.source_split.eq('train'), 'class_id'].unique().tolist())
            experiment.directory = Path(folder)
            experiment.hard_features = np.zeros((len(records), 48))
            experiment.query_mask = np.ones(len(records), dtype=bool)
            context = experiment.context('official')
            query = int(context['test_pairs'].query_row.iloc[0])
            experiment.query_mask[query] = False
            with self.assertRaisesRegex(ValueError, 'coverage mask'):
                experiment.context('official')

if __name__ == '__main__':
    unittest.main()
