"""Audit the returned F5 families against round 1 and export model-free evidence (mirrors tools/audit_result_archive.py, which expects exactly four single-feature groups)."""
import argparse, hashlib, io, json, sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS = ('accuracy', 'precision', 'TPR', 'FPR', 'F1', 'balanced_accuracy', 'ROC_AUC', 'average_precision')
KEYS = ['fold', 'algorithm', 'features', 'condition', 'negative_subset']
P = argparse.ArgumentParser(); P.add_argument('--archive', type=Path); P.add_argument('--output', type=Path); P.add_argument('--source-code', type=Path)
a = P.parse_args()
names_src = ('sop_baselines.py', 'sop_supcon.py', 'sop_reporting.py', 'sop_config.py', 'sop_feature_bank.py',
             'sop_features.py', 'sop_protocol.py', 'sop_experiment.py', 'sop_calibration.py', 'sop_tuning.py')
h = hashlib.sha256()
for n in names_src: h.update((a.source_code / 'utils' / n).read_bytes())
fingerprint = h.hexdigest()
prefix = 'results/Main_Experiments/'; group = 'F5_Traditional_Content_HSV_Gradient'
with ZipFile(a.archive) as z:
    names = {i.filename for i in z.infolist() if not i.is_dir()}
    rj = lambda n: json.loads(z.read(n))
    rc = lambda n: pd.read_csv(io.BytesIO(z.read(n)), compression='gzip' if n.endswith('.gz') else None)
    # Reference test-pair hashes from round-1 F1 predictions in the same package.
    ref = {}
    for fold in range(1, 11):
        p = rc(f'{prefix}F1_Traditional_HSV48/baselines/outer_{fold:02d}/LR_HSV48/predictions_clean.csv.gz')
        cols = ['query_image_id', 'reference_image_id', 'label', 'kind', 'query_product_id', 'reference_product_id']
        ref[fold] = hashlib.sha256(p[cols].to_csv(index=False).encode()).hexdigest()
    families, summaries, all_metrics, ids, checked = [], [], [], [], 0
    r1_protocol = rj(prefix + 'F1_Traditional_HSV48/baselines/protocol.json')['shared_protocol_id']
    for family in ('baselines', 'supcon'):
        base = f'{prefix}{group}/{family}/'
        identity = rj(base + 'protocol.json'); c = identity['config']
        assert identity['code_fingerprint'] == fingerprint, 'fingerprint mismatch'
        assert identity['shared_protocol_id'] == r1_protocol
        assert c['outer_folds'] == 10 and c['inner_folds'] == 3 and c['mode'] == 'formal' and not c['run_official_test']
        metrics, summary = rc(base + 'outer_fold_metrics.csv'), rc(base + 'summary_mean_sd.csv')
        algorithms = ['LR', 'SVM'] if family == 'baselines' else ['SupCon']
        features = list(c['feature_sets'])
        assert len(metrics) == len(algorithms) * len(features) * 10 * 5 * 3
        assert not metrics.duplicated(KEYS).any() and metrics.groupby(KEYS[1:]).fold.nunique().eq(10).all()
        for fold in range(1, 11):
            fp = base + f'outer_{fold:02d}/'
            saved = pd.DataFrame(rj(fp + 'completed.json')['rows']).sort_values(KEYS).reset_index(drop=True)
            actual = metrics[metrics.fold.eq(fold)].sort_values(KEYS).reset_index(drop=True)
            assert saved[KEYS].equals(actual[KEYS]); np.testing.assert_allclose(saved[list(METRICS)], actual[list(METRICS)], atol=1e-12)
            for alg in algorithms:
                for feat in features:
                    folder = f'{fp}{alg}_{feat}/'; cal = rj(folder + 'calibration.json')
                    fit, calp = set(cal['fitting_products']), set(cal['calibration_products']); assert not fit & calp
                    for cond in c['conditions']:
                        pr = rc(folder + f'predictions_{cond}.csv.gz')
                        cols = ['query_image_id', 'reference_image_id', 'label', 'kind', 'query_product_id', 'reference_product_id']
                        assert hashlib.sha256(pr[cols].to_csv(index=False).encode()).hexdigest() == ref[fold], 'pairs differ from R1'
                        assert ((pr.score >= pr.threshold).astype(int) == pr.prediction).all()
                        assert not (set(pr.query_product_id) | set(pr.reference_product_id)) & (fit | calp)
                        np.testing.assert_allclose(pr.threshold, cal['threshold'], atol=1e-12)
                        for sub in ('all', 'random', 'colour_hard'):
                            f = pr if sub == 'all' else pr[pr.kind.isin(['positive', sub])]
                            y, pd_ = f.label.to_numpy(), f.prediction.to_numpy()
                            tp = int(((y == 1) & (pd_ == 1)).sum()); fpn = int(((y == 0) & (pd_ == 1)).sum())
                            tn = int(((y == 0) & (pd_ == 0)).sum()); fn = int(((y == 1) & (pd_ == 0)).sum())
                            rec, fpr = tp / (tp + fn), fpn / (fpn + tn)
                            exp = {'TPR': rec, 'FPR': fpr, 'F1': 2 * tp / (2 * tp + fpn + fn), 'accuracy': (tp + tn) / len(y),
                                   'balanced_accuracy': (rec + 1 - fpr) / 2, 'ROC_AUC': roc_auc_score(y, f.score),
                                   'average_precision': average_precision_score(y, f.score)}
                            row = metrics[metrics.fold.eq(fold) & metrics.algorithm.eq(alg) & metrics.features.eq(feat)
                                          & metrics.condition.eq(cond) & metrics.negative_subset.eq(sub)].iloc[0]
                            for k, v in exp.items(): np.testing.assert_allclose(row[k], v, atol=1e-10)
                        checked += 1
        for row in summary.to_dict('records'):
            v = metrics[metrics.algorithm.eq(row['algorithm']) & metrics.features.eq(row['features'])
                        & metrics.condition.eq(row['condition']) & metrics.negative_subset.eq(row['negative_subset'])][row['metric']]
            np.testing.assert_allclose(row['mean'], v.mean(), atol=1e-12); np.testing.assert_allclose(row['sd_across_outer_folds'], v.std(ddof=1), atol=1e-12)
        metrics['group'] = summary['group'] = group
        all_metrics.append(metrics); summaries.append(summary); ids.append(identity)
        families.append({'group': group, 'family': family, 'features': features, 'metric_rows': len(metrics), 'summary_rows': len(summary),
                         'completed_folds': 10, 'torch': identity.get('torch', '-')})
    # Recompute every delta row that involves F5 against the round-1 fold metrics.
    others = []
    for proto in sorted(n for n in names if n.endswith('/protocol.json') and group not in n):
        b = proto.rsplit('/', 1)[0] + '/'; m = rc(b + 'outer_fold_metrics.csv'); m['group'] = b.split('/')[2]; others.append(m)
    full = pd.concat(all_metrics + others, ignore_index=True).set_index(['group', 'algorithm', 'features', 'condition', 'negative_subset', 'fold'])
    deltas = rc(prefix + 'comparisons/paired_group_deltas.csv')
    f5 = deltas[deltas.added_group.eq(group) | deltas.baseline_group.eq(group)]
    for r in f5.to_dict('records'):
        k = (r['algorithm'], None, r['condition'], r['negative_subset'], r['fold'])
        add = full.loc[(r['added_group'], r['algorithm'], r['added_features'], r['condition'], r['negative_subset'], r['fold']), r['metric']]
        bas = full.loc[(r['baseline_group'], r['algorithm'], r['baseline_features'], r['condition'], r['negative_subset'], r['fold']), r['metric']]
        np.testing.assert_allclose(r['delta_added_minus_baseline'], add - bas, atol=1e-12)
    assert not rj(prefix + 'comparisons/skipped_incompatible_comparisons.json')
    manifest, excluded = [], []
    for name in sorted(names):
        path = PurePosixPath(name); rel = name[len(prefix):]
        if not (rel.startswith(group + '/') or rel.startswith('comparisons/')): continue
        assert not path.is_absolute() and '..' not in path.parts
        if path.suffix not in ('.csv', '.gz', '.json', '.png'): excluded.append(rel); continue
        content = z.read(name); src = hashlib.sha256(content).hexdigest(); red = False
        if path.name == 'protocol.json':
            d = json.loads(content); d['config'].update(data_dir='processed_sop', protocol_dir='protocols/Main_Experiments_Organized', cache_dir='feature_cache', results_dir='results')
            content = (json.dumps(d, indent=2) + '\n').encode(); red = True
        t = a.output / 'artifacts' / rel; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(content)
        manifest.append({'path': rel, 'source_sha256': src, 'published_sha256': hashlib.sha256(content).hexdigest(), 'path_redacted': red})
    summary = pd.concat(summaries, ignore_index=True); summary.to_csv(a.output / 'all_metrics.csv', index=False)
    summary[summary.condition.isin(['clean', 'occlusion25']) & summary.metric.isin(['TPR', 'FPR', 'ROC_AUC'])].to_csv(a.output / 'key_metrics.csv', index=False)
    rec = {'date': '2026-10-12', 'round': 'R1 (supplement, same run name, code and protocol as R1)',
           'source': 'User-returned results_F5.zip; audited on the development Mac; no retraining here',
           'families': families, 'test_prediction_files_recomputed': checked, 'fold_metric_rows': int(sum(len(m) for m in all_metrics)),
           'summary_rows': len(summary), 'f5_paired_delta_rows_verified': len(f5), 'summary_mean_and_sample_sd_verified': True,
           'completed_records_match_metrics': True, 'saved_product_partition_disjointness_checked': True,
           'test_pairs_identical_to_round1_F1_per_fold': True, 'code_fingerprint': fingerprint,
           'code_fingerprint_source': 'repository commit c275cd4 PythonCode/utils (Windows line endings)',
           'shared_protocol_id': ids[0]['shared_protocol_id'], 'published_files': len(manifest), 'excluded_model_files': excluded,
           'limits': ['F5 SupCon ran with CPU torch 2.14.1+cpu; F1/F2 SupCon used CUDA (cu130).',
                      'Colour-hard negatives were selected with clean content-only HSV48, the F5 ContentHSV48 descriptor itself.',
                      'No statistical inference, independent-seed repeats or official holdout.']}
    for fn, data in (('audit.json', rec), ('file_manifest.json', manifest)):
        (a.output / fn).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: rec[k] for k in ('test_prediction_files_recomputed', 'fold_metric_rows', 'summary_rows', 'f5_paired_delta_rows_verified', 'published_files')}), len(excluded), 'model files excluded')
