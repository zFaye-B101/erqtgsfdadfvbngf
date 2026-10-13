"""Verify read-only source use and strict no-overwrite recalibration."""
import importlib.util
import json
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.sop_protocol import manual_metrics

spec = importlib.util.spec_from_file_location('recalibration', ROOT / '03_SOP_Recalibrate_90.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
ARTIFACTS = ROOT / '.verification_artifacts'
ARTIFACTS.mkdir(exist_ok=True)


class RecalibrationTests(unittest.TestCase):
    def test_directory_and_zip_preserve_source_and_reject_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder:
            root = Path(folder)
            source = root / 'results'
            family = source / 'Example' / 'F1' / 'baselines'
            family.mkdir(parents=True)
            config = dict(target_validation_recall=.95, outer_folds=2, mode='smoke',
                          conditions={'clean': {}}, feature_sets={'HSV48': {}}, comparisons=[])
            (family / 'protocol.json').write_text(json.dumps({'config': config}), encoding='utf-8')
            rows = []
            for fold in [1, 2]:
                model = family / f'outer_{fold:02d}' / 'LR_HSV48'
                model.mkdir(parents=True)
                pred = pd.DataFrame({'label': [1]*20+[0]*20+[0]*20,
                    'kind': ['positive']*20+['random']*20+['colour_hard']*20,
                    'score': np.r_[np.linspace(0, 1, 20), np.linspace(-.1,.7,20), np.linspace(.1,.9,20)],
                    'threshold': .05})
                pred.to_csv(model / 'calibration_predictions.csv.gz', index=False)
                pred.to_csv(model / 'predictions_clean.csv.gz', index=False)
                for subset in ['all','random','colour_hard']:
                    part = pred if subset=='all' else pred[pred.kind.isin(['positive',subset])]
                    rows.append(dict(fold=fold,algorithm='LR',features='HSV48',condition='clean',negative_subset=subset,
                                     **manual_metrics(part.label.to_numpy(),part.score.to_numpy(),.05)))
            pd.DataFrame(rows).to_csv(family / 'outer_fold_metrics.csv',index=False)
            # Verify directory discovery keeps all eight groups after splitting F5.
            group_names = ['F1', 'F1.1', 'F2', 'F2.1', 'F3', 'F4', 'F4.1', 'F4.2']
            for group in group_names[1:]:
                shutil.copytree(family, source / 'Example' / group / 'baselines')
            before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            output = root / 'results_recall90'
            entry.reprocess(source,output)
            result = pd.read_csv(output / 'Example/F1/baselines/outer_fold_metrics.csv')
            self.assertEqual(sorted(p.parent.parent.name for p in output.glob('Example/*/baselines/outer_fold_metrics.csv')),
                             sorted(group_names))
            self.assertTrue(result.TPR.eq(.9).all())
            self.assertEqual(before,{p.relative_to(source).as_posix():p.read_bytes() for p in source.rglob('*') if p.is_file()})
            result_bytes = (output / 'Example/F1/baselines/outer_fold_metrics.csv').read_bytes()
            with self.assertRaises(FileExistsError):
                entry.reprocess(source,output)
            self.assertEqual(result_bytes,(output / 'Example/F1/baselines/outer_fold_metrics.csv').read_bytes())
            archive = root / 'results.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for name in before:
                    z.write(source / name,'results/'+name)
            entry.reprocess(archive, root / 'zip_output')
            zipped = pd.read_csv(root / 'zip_output/Example/F1/baselines/outer_fold_metrics.csv')
            pd.testing.assert_frame_equal(result,zipped)
            with self.assertRaises(ValueError):
                entry.reprocess(source, source / 'nested_output')


if __name__ == '__main__':
    unittest.main()
