"""Local feature checks with hand calculations and random weights, no downloads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.sop_config import discover_experiments, load_config
from utils.sop_feature_bank import EXTRACTORS, ResNet18Extractor, ResNet18CropMaxExtractor, build_banks


def plan_a_config():
    return load_config(ROOT / 'config.json', experiment_path=
                       ROOT / 'experiment_configs/F4.1_ResNet_GlobalRegionMax1024.json')


class LocalFeatureTests(unittest.TestCase):
    def test_plan_a_preserves_global_and_ignores_region_order(self):
        values = np.zeros((2, 2560), dtype=np.float32)
        values[:, :512] = np.arange(512)
        regions = values[:, 512:].reshape(2, 4, 512)
        regions[:, :, 0] = [0.9, 0.1, 0.2, 0.1]
        regions[:, :, 1] = [0.2, 0.6, 0.4, 0.3]
        config = plan_a_config()
        def transform(source):
            with patch('utils.sop_feature_bank.extract_cached', return_value=source) as extract:
                result = next(iter(build_banks(None, None, config, 'synthetic', None).values()))['clean']
                self.assertEqual(extract.call_count, len(config['conditions']))
                return result
        result = transform(values)
        self.assertEqual(result.shape, (2, 1024))
        np.testing.assert_array_equal(result[:, :512], values[:, :512])
        np.testing.assert_allclose(result[:, 512:514], [[.9, .6], [.9, .6]])
        permuted = np.concatenate([values[:, :512], regions[:, [2, 0, 3, 1]].reshape(2, -1)], axis=1)
        np.testing.assert_array_equal(transform(permuted), result)
        self.assertFalse(transform(np.zeros((1, 2560))).any())

    def test_crop_geometry_and_hand_computed_aggregation(self):
        coordinates = np.zeros((224, 224, 3), dtype=np.uint8)
        coordinates[:, :, 0] = np.arange(224)[None, :]
        coordinates[:, :, 1] = np.arange(224)[:, None]
        image = Image.fromarray(coordinates)
        calls = []
        def fake_init(instance, settings):
            instance.metadata = {'extractor': 'resnet18'}
        def fake_forward(instance, images):
            calls.append([value.copy() for value in images])
            means = np.array([np.asarray(value).mean(axis=(0, 1))[:2] for value in images])
            result = np.zeros((len(images), 2560), dtype=np.float32)
            result[:, :2] = means
            return result
        with patch.object(ResNet18Extractor, '__init__', fake_init), patch.object(ResNet18Extractor, '__call__', fake_forward):
            extractor = ResNet18CropMaxExtractor({'device': 'cpu'})
            result = extractor([image])
            self.assertEqual(len(calls), 6)
            np.testing.assert_allclose(result[0, :2], [111.5, 111.5])
            np.testing.assert_allclose(result[0, 512:514], [143.5, 143.5], atol=.01)
            for box, actual in zip(extractor.crop_boxes, calls[1:]):
                expected = image.crop(box).resize((224, 224), Image.Resampling.BILINEAR)
                np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(expected))
            with self.assertRaises(ValueError):
                extractor([Image.new('RGB', (160, 160))])

    def test_crop_random_weight_forward_matches_global_and_crops(self):
        import torch
        import torchvision.models as models
        torch.set_num_threads(2)
        model = models.resnet18(weights=None)
        with patch('torchvision.models.resnet18', return_value=model):
            base = ResNet18Extractor({'device': 'cpu'})
            cropped = ResNet18CropMaxExtractor({'device': 'cpu'})
        image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (224, 224, 3), dtype=np.uint8))
        expected_local = np.maximum.reduce([base([image.crop(box).resize((224, 224), Image.Resampling.BILINEAR)])[:, :512]
                                            for box in cropped.crop_boxes])
        result = cropped([image])
        self.assertEqual(result.shape, (1, 1024))
        np.testing.assert_array_equal(result[:, :512], base([image])[:, :512])
        np.testing.assert_array_equal(result[:, 512:], expected_local)
        self.assertTrue(np.isfinite(result).all())
        self.assertFalse(any(parameter.requires_grad for parameter in cropped.model.parameters()))

    def test_plan_a_shares_source_cache_and_rejects_tampering(self):
        from utils import sop_feature_bank as bank
        calls = []
        class FakeSource:
            dimension = 2560
            def __init__(self, settings):
                self.metadata = {'extractor': 'resnet18', 'test_weights': 'synthetic'}
            def __call__(self, images):
                calls.append(len(images))
                return np.tile(np.arange(2560, dtype=np.float32), (len(images), 1))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (224, 224), (80, 90, 100)).save(root / 'image.jpg')
            frame = pd.DataFrame([{'image_id': 1, 'original_width': 224, 'original_height': 224,
                                   'processed_relative_path': 'image.jpg'}])
            config = plan_a_config()
            config.update(cache_dir=str(root / 'cache'), seed=1,
                          extraction={'device': 'cpu', 'batch_size': 1},
                          conditions={'clean': {}, 'occlusion25': {'area_fraction': .25, 'fill': 127}})
            mask = np.array([True])
            with patch.dict(EXTRACTORS, {'resnet18': FakeSource}), patch.dict(bank._INSTANCES, {}, clear=True):
                result = next(iter(build_banks(frame, root, config, 'synthetic', mask).values()))
                self.assertEqual(len(calls), 2)  # one source batch per condition, shared by both components
                values = np.arange(2560, dtype=np.float32)[None]
                expected = np.concatenate([values[:, :512], values[:, 512:].reshape(1, 4, 512).max(axis=1)], axis=1)
                for actual in result.values():
                    np.testing.assert_array_equal(actual, expected)
                build_banks(frame, root, config, 'synthetic', mask)
                self.assertEqual(len(calls), 2)
                metadata = next((root / 'cache/resnet18').glob('*/metadata.json'))
                record = json.loads(metadata.read_text())
                record['seed'] = 999
                metadata.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    build_banks(frame, root, config, 'synthetic', mask)

    def test_canonical_groups_and_preflight(self):
        import runpy
        groups = discover_experiments(ROOT / 'experiment_configs')
        self.assertIn('F6_ResNet50_Global2048', groups)
        self.assertIn('F4.1_ResNet_GlobalRegionMax1024', groups)
        self.assertNotIn('F7_ResNet18_RegionMax1024', groups)
        self.assertNotIn('F8_ResNet18_CropMax1024', groups)
        self.assertNotIn('resnet18_region_max', EXTRACTORS)
        config = load_config(ROOT / 'config.json',
                             experiment_path=groups['F4.2_ResNet_GlobalCropMax1024'][0])
        spec = next(iter(config['feature_sets'].values()))
        self.assertEqual(spec['extractor'], 'resnet18_crop_max')
        self.assertEqual(spec['columns'], [0, 1024])
        self.assertEqual(set(config['conditions']), {'clean', 'occlusion25', 'noise005', 'blur15', 'rotation15'})
        preflight = runpy.run_path(str(ROOT / '00_Check_Environment.py'))
        self.assertEqual(preflight['RESNET_WEIGHTS']['resnet18_crop_max'], preflight['RESNET_WEIGHTS']['resnet18'])


if __name__ == '__main__':
    unittest.main()
