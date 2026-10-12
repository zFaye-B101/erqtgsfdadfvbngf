"""Verify returned SOP metrics and export portable, model-free evidence.

Inputs: a results ZIP and the matching PythonCode source tree.
Output: audited CSV/JSON/PNG evidence, file hashes and an audit record.
Original archives are never changed; pickle and torch checkpoints are not loaded.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS = ('accuracy', 'precision', 'TPR', 'FPR', 'F1', 'balanced_accuracy',
           'ROC_AUC', 'average_precision')
KEYS = ['fold', 'algorithm', 'features', 'condition', 'negative_subset']


def audit(archive, output, source_code):
    """Check predictions, checkpoints, summaries and cross-group comparisons."""
    sys.path.insert(0, str(source_code.resolve()))
    from utils.sop_feature_bank import file_digest
    source_names = ('sop_baselines.py', 'sop_supcon.py', 'sop_reporting.py', 'sop_config.py',
                    'sop_feature_bank.py', 'sop_features.py', 'sop_protocol.py',
                    'sop_experiment.py', 'sop_calibration.py', 'sop_tuning.py')
    fingerprint = file_digest([source_code / 'utils' / name for name in source_names])
    families, summaries, all_metrics, identities = [], [], [], []
    checked = 0
    test_pair_hashes = {}
    with ZipFile(archive) as zipped:
        names = {item.filename for item in zipped.infolist() if not item.is_dir()}
        prefix = 'results/Main_Experiments/'
        def read_json(name):
            return json.loads(zipped.read(name))
        def read_csv(name):
            return pd.read_csv(io.BytesIO(zipped.read(name)),
                               compression='gzip' if name.endswith('.gz') else None)
        protocols = sorted(name for name in names if name.endswith('/protocol.json'))
        assert len(protocols) == 8, 'Expected four groups and two algorithm families.'
        for protocol_path in protocols:
            base = protocol_path.rsplit('/', 1)[0] + '/'
            identity = read_json(protocol_path)
            group, family = base.split('/')[2:4]
            config = identity['config']
            assert identity['code_fingerprint'] == fingerprint
            assert config['outer_folds'] == 10 and config['mode'] == 'formal'
            assert not config['run_official_test']
            metrics = read_csv(base + 'outer_fold_metrics.csv')
            summary = read_csv(base + 'summary_mean_sd.csv')
            algorithms = ['LR', 'SVM'] if family == 'baselines' else ['SupCon']
            feature = next(iter(config['feature_sets']))
            assert len(config['feature_sets']) == 1
            assert len(metrics) == len(algorithms) * 10 * 5 * 3
            assert len(summary) == len(algorithms) * 5 * 3 * len(METRICS)
            assert set(metrics.algorithm) == set(algorithms)
            assert not metrics.duplicated(KEYS).any()
            assert metrics.groupby(KEYS[1:]).fold.nunique().eq(10).all()
            for fold in range(1, 11):
                fold_path = base + f'outer_{fold:02d}/'
                saved = pd.DataFrame(read_json(fold_path + 'completed.json')['rows'])
                saved = saved.sort_values(KEYS).reset_index(drop=True)
                actual = metrics.loc[metrics.fold.eq(fold)].sort_values(KEYS).reset_index(drop=True)
                assert saved[KEYS].equals(actual[KEYS])
                np.testing.assert_allclose(saved[list(METRICS)], actual[list(METRICS)], atol=1e-12)
                for algorithm in algorithms:
                    folder = fold_path + algorithm + '_' + feature + '/'
                    calibration = read_json(folder + 'calibration.json')
                    fitting = set(calibration['fitting_products'])
                    calibrating = set(calibration['calibration_products'])
                    assert not fitting & calibrating
                    for condition in config['conditions']:
                        predictions = read_csv(folder + 'predictions_' + condition + '.csv.gz')
                        pair_columns = ['query_image_id', 'reference_image_id', 'label', 'kind',
                                        'query_product_id', 'reference_product_id']
                        pair_hash = hashlib.sha256(predictions[pair_columns].to_csv(index=False).encode('utf-8')).hexdigest()
                        assert test_pair_hashes.setdefault(fold, pair_hash) == pair_hash, 'Test pairs differ between runs.'
                        assert np.isfinite(predictions.score).all()
                        assert ((predictions.score >= predictions.threshold).astype(int)
                                == predictions.prediction).all()
                        products = set(predictions.query_product_id) | set(predictions.reference_product_id)
                        assert not products & (fitting | calibrating)
                        assert (predictions.query_product_id.eq(predictions.reference_product_id)
                                == predictions.label.eq(1)).all()
                        np.testing.assert_allclose(predictions.threshold, calibration['threshold'], atol=1e-12)
                        for subset in ('all', 'random', 'colour_hard'):
                            frame = predictions if subset == 'all' else predictions.loc[
                                predictions.kind.isin(['positive', subset])]
                            y, pred = frame.label.to_numpy(), frame.prediction.to_numpy()
                            tp = int(((y == 1) & (pred == 1)).sum())
                            fp = int(((y == 0) & (pred == 1)).sum())
                            tn = int(((y == 0) & (pred == 0)).sum())
                            fn = int(((y == 1) & (pred == 0)).sum())
                            recall, fpr = tp / (tp + fn), fp / (fp + tn)
                            expected = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                                        'accuracy': (tp + tn) / len(y),
                                        'precision': tp / (tp + fp) if tp + fp else 0,
                                        'TPR': recall, 'FPR': fpr, 'F1': 2 * tp / (2 * tp + fp + fn),
                                        'balanced_accuracy': (recall + 1 - fpr) / 2,
                                        'ROC_AUC': roc_auc_score(y, frame.score),
                                        'average_precision': average_precision_score(y, frame.score)}
                            row = metrics.loc[metrics.fold.eq(fold) & metrics.algorithm.eq(algorithm)
                                              & metrics.condition.eq(condition)
                                              & metrics.negative_subset.eq(subset)].iloc[0]
                            for metric, value in expected.items():
                                np.testing.assert_allclose(row[metric], value, atol=1e-10,
                                                           err_msg=f'{group}/{fold}/{algorithm}/{condition}/{subset}/{metric}')
                            assert row.pair_count == len(frame)
                        checked += 1
            for row in summary.to_dict('records'):
                values = metrics.loc[metrics.algorithm.eq(row['algorithm']) & metrics.features.eq(row['features'])
                                     & metrics.condition.eq(row['condition'])
                                     & metrics.negative_subset.eq(row['negative_subset'])][row['metric']]
                np.testing.assert_allclose(row['mean'], values.mean(), atol=1e-12)
                np.testing.assert_allclose(row['sd_across_outer_folds'], values.std(ddof=1), atol=1e-12)
                assert row['outer_fold_count'] == 10
            metrics['group'], summary['group'] = group, group
            all_metrics.append(metrics)
            summaries.append(summary)
            identities.append(identity)
            families.append({'group': group, 'family': family, 'metric_rows': len(metrics),
                             'summary_rows': len(summary), 'completed_folds': 10})
        assert len({item['shared_protocol_id'] for item in identities}) == 1
        for group in {item['group'] for item in families}:
            pair = [item for item in identities if item['config']['feature_profile'] == group]
            assert pair[0]['config'] == pair[1]['config']
            combined = read_csv(prefix + group + '/combined_summary_mean_sd.csv')
            actual = pd.concat([frame.drop(columns='group') for frame in summaries
                                if frame.group.iloc[0] == group], ignore_index=True)
            keys = ['algorithm', 'features', 'condition', 'negative_subset', 'metric']
            pd.testing.assert_frame_equal(combined.sort_values(keys).reset_index(drop=True),
                                          actual.sort_values(keys).reset_index(drop=True))
        full_metrics = pd.concat(all_metrics, ignore_index=True)
        deltas = read_csv(prefix + 'comparisons/paired_group_deltas.csv')
        assert len(deltas) == 10800
        values = full_metrics.set_index(['group', 'algorithm', 'condition', 'negative_subset', 'fold'])
        for row in deltas.to_dict('records'):
            key = tuple(row[name] for name in ['algorithm', 'condition', 'negative_subset', 'fold'])
            difference = values.loc[(row['added_group'],) + key, row['metric']] - values.loc[
                (row['baseline_group'],) + key, row['metric']]
            np.testing.assert_allclose(row['delta_added_minus_baseline'], difference, atol=1e-12)
        assert not read_json(prefix + 'comparisons/skipped_incompatible_comparisons.json')

        # Export approved result evidence without executable models or host paths.
        manifest, excluded = [], []
        for name in sorted(names):
            path = PurePosixPath(name)
            assert name.startswith(prefix) and not path.is_absolute() and '..' not in path.parts
            if path.suffix not in ('.csv', '.gz', '.json', '.png'):
                excluded.append(name[len(prefix):])
                continue
            content = zipped.read(name)
            original_hash = hashlib.sha256(content).hexdigest()
            redacted = False
            if path.name == 'protocol.json':
                data = json.loads(content)
                data['config'].update(data_dir='processed_sop',
                                      protocol_dir='protocols/Main_Experiments_Organized',
                                      cache_dir='feature_cache', results_dir='results')
                content = (json.dumps(data, indent=2) + '\n').encode('utf-8')
                redacted = True
            relative = name[len(prefix):]
            target = output / 'artifacts' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            manifest.append({'path': relative, 'source_sha256': original_hash,
                             'published_sha256': hashlib.sha256(content).hexdigest(),
                             'path_redacted': redacted})
        summary = pd.concat(summaries, ignore_index=True)
        summary.to_csv(output / 'all_metrics.csv', index=False)
        focus = summary.loc[summary.condition.isin(['clean', 'occlusion25'])
                            & summary.negative_subset.eq('colour_hard')
                            & summary.metric.isin(['TPR', 'FPR', 'ROC_AUC'])]
        focus.to_csv(output / 'key_metrics.csv', index=False)
        audit_record = {'date': '2026-10-12', 'source': 'User-returned GPU-computer results; no formal retraining here',
                        'families': families, 'test_prediction_files_recomputed': checked,
                        'fold_metric_rows': len(full_metrics), 'summary_rows': len(summary),
                        'paired_delta_rows_verified': len(deltas),
                        'summary_mean_and_sample_sd_verified': True,
                        'completed_records_match_metrics': True,
                        'saved_product_partition_disjointness_checked': True,
                        'identical_test_pairs_across_groups_algorithms_and_conditions': True,
                        'current_model_source_fingerprint_matches': True,
                        'code_fingerprint': fingerprint,
                        'shared_protocol_id': identities[0]['shared_protocol_id'],
                        'published_files': len(manifest), 'excluded_model_files': excluded,
                        'limits': ['Full preprocessing manifest and product-fold protocol were not returned.',
                                   'All inner partition memberships cannot be independently audited from these outputs.',
                                   'F5 and official holdout results are absent.',
                                   'No statistical inference or independent-seed repeats are claimed.']}
        for filename, data in [('audit.json', audit_record), ('file_manifest.json', manifest)]:
            (output / filename).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: audit_record[key] for key in ('test_prediction_files_recomputed',
                          'fold_metric_rows', 'summary_rows', 'paired_delta_rows_verified', 'published_files')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-code', type=Path, default=Path(__file__).resolve().parents[1] / 'PythonCode')
    arguments = parser.parse_args()
    audit(arguments.archive, arguments.output, arguments.source_code)
