"""Reproduce COMP90051 Lecture 15 comparisons from archived fold metrics."""

import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import scipy
from scipy.special import betainc, betaincinv

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE95 = ROOT / 'docs/experiments/2026-10-13-R2-local-features'
SOURCE90 = ROOT / 'docs/experiments/2026-10-13-R1-R2-recall90'
RUN = 'ResNet_LocalFeature_Comparison'
GROUPS = ['F3', 'F4', 'F4.1', 'F4.2']
ALGORITHMS = ['LR', 'SVM', 'SupCon']
CONDITIONS = ['clean', 'occlusion25', 'noise005', 'blur15', 'rotation15']
SUBSETS = ['all', 'random', 'colour_hard']


def t_sf(x, df):
    """Student upper-tail probability from the regularized incomplete beta."""
    tail = 0.5 * float(betainc(df / 2, 0.5, df / (df + x * x)))
    return tail if x >= 0 else 1 - tail


def t_critical(alpha, df):
    z = float(betaincinv(df / 2, 0.5, 2 * alpha))
    return math.sqrt(df * (1 / z - 1))


def lecture_test(a, b):
    """H1: mean(a) > mean(b); lecture population variance and ceiling df."""
    assert len(a) == len(b) == 10 or len(a) == len(b) == 9
    n = len(a)
    va, vb = statistics.pvariance(a), statistics.pvariance(b)
    difference = statistics.mean(a) - statistics.mean(b)
    if va + vb == 0:
        return {'statistic': None, 'df_raw': None, 'df': None,
                'critical': None, 'p': None, 'reject': False}
    se = math.sqrt((va + vb) / n)
    statistic = difference / se
    df_raw = (va + vb) ** 2 * (n - 1) / (va * va + vb * vb)
    df = math.ceil(df_raw)
    critical = t_critical(0.05, df)
    return {'statistic': statistic, 'df_raw': df_raw, 'df': df,
            'critical': critical, 'p': t_sf(statistic, df),
            'reject': statistic > critical}


def read_table(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        if row['run'] != RUN:
            continue
        key = (row['group'].split('_')[0], row['algorithm'], row['condition'],
               row['negative_subset'], int(row['fold']))
        assert key not in result, key
        for metric in ['TPR', 'FPR', 'ROC_AUC']:
            row[metric] = float(row[metric])
            assert math.isfinite(row[metric]) and 0 <= row[metric] <= 1
        tp, fp, tn, fn = [int(row[k]) for k in ['TP', 'FP', 'TN', 'FN']]
        assert tp + fn == 192
        assert fp + tn == (384 if row['negative_subset'] == 'all' else 192)
        assert abs(row['TPR'] - tp / (tp + fn)) < 1e-12
        assert abs(row['FPR'] - fp / (fp + tn)) < 1e-12
        result[key] = row
    assert len(result) == 1800
    return result


def save_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    example1 = lecture_test([11, 7, 13, 12, 12, 9, 10, 7, 10],
                            [10, 8, 12, 10, 11, 9, 13, 7, 9])
    example2 = lecture_test([10, 12, 14, 13, 13, 10, 11, 10, 11],
                            [10, 8, 12, 10, 11, 9, 13, 7, 9])
    assert not example1['reject'] and example2['reject']
    assert example1['df'] == example2['df'] == 16
    tables = {95: read_table(SOURCE95 / 'outer_fold_metrics.csv'),
              90: read_table(SOURCE90 / 'outer_fold_metrics.csv')}
    for key in tables[95]:
        assert abs(tables[95][key]['ROC_AUC'] - tables[90][key]['ROC_AUC']) < 1e-12

    # Product memberships, not pair counts, establish separation of partitions.
    partitions = {}
    calibration_files = list((SOURCE95 / 'artifacts').glob('**/calibration.json'))
    assert len(calibration_files) == 120
    for path in calibration_files:
        fold = int(next(x for x in path.parts if x.startswith('outer_')).split('_')[1])
        info = json.loads(path.read_text())
        fit, cal = set(info['fitting_products']), set(info['calibration_products'])
        with gzip.open(path.parent / 'predictions_clean.csv.gz', 'rt') as stream:
            pairs = list(csv.DictReader(stream))
        # Column names are verified before running this script.
        test = {int(r['query_product_id']) for r in pairs}
        test |= {int(r['reference_product_id']) for r in pairs}
        assert len(test) == 192 and len(fit) == 1380 and len(cal) == 348
        assert not fit & cal and not fit & test and not cal & test
        signature = (fit, cal, test)
        if fold in partitions:
            assert signature == partitions[fold]
        partitions[fold] = signature
    assert len(partitions) == 10
    assert len(set.union(*(p[2] for p in partitions.values()))) == 1920
    universe = set.union(*partitions[1])
    assert all(set.union(*p) == universe for p in partitions.values())

    results, differences, means = [], [], []
    for target, table in tables.items():
        for algorithm in ALGORITHMS:
            for condition in CONDITIONS:
                for subset in SUBSETS:
                    vectors = {group: {metric: [table[(group, algorithm, condition, subset, f)][metric]
                                                for f in range(1, 11)]
                                       for metric in ['FPR', 'TPR', 'ROC_AUC']}
                               for group in GROUPS}
                    for group in GROUPS:
                        for metric, values in vectors[group].items():
                            mu = statistics.mean(values)
                            half_width = t_critical(0.025, 9) * math.sqrt(statistics.pvariance(values) / 10)
                            means.append(dict(target=target, group=group, algorithm=algorithm,
                                              condition=condition, subset=subset, metric=metric,
                                              mean=mu, sample_sd=statistics.stdev(values),
                                              lecture_variance=statistics.pvariance(values),
                                              lecture_pointwise95_low=mu-half_width,
                                              lecture_pointwise95_high=mu+half_width))
                    for group in GROUPS[1:]:
                        for metric in ['FPR', 'TPR', 'ROC_AUC']:
                            base, local = vectors['F3'][metric], vectors[group][metric]
                            deltas = [b - a for a, b in zip(base, local)]
                            # The directions express the questions, not observed signs.
                            first, second = (local, base) if metric == 'ROC_AUC' else (base, local)
                            result = lecture_test(first, second)
                            row = dict(target=target, baseline='F3', local=group,
                                       algorithm=algorithm, condition=condition, subset=subset,
                                       metric=metric,
                                       hypothesis={'FPR': 'local_FPR_lower', 'TPR': 'local_TPR_lower',
                                                   'ROC_AUC': 'local_AUC_higher'}[metric],
                                       baseline_mean=statistics.mean(base), local_mean=statistics.mean(local),
                                       local_minus_baseline=statistics.mean(deltas),
                                       paired_sample_sd=statistics.stdev(deltas),
                                       decreased_folds=sum(d < 0 for d in deltas),
                                       increased_folds=sum(d > 0 for d in deltas), **result)
                            results.append(row)
                            for fold, (a, b, d) in enumerate(zip(base, local, deltas), 1):
                                differences.append(dict(target=target, local=group, algorithm=algorithm,
                                                        condition=condition, subset=subset, metric=metric,
                                                        fold=fold, baseline=a, local_value=b, difference=d))
    assert len(results) == 810 and len(differences) == 8100
    save_csv('all_comparisons.csv', results)
    save_csv('fold_differences.csv', differences)
    save_csv('model_means_and_lecture_intervals.csv', means)
    primary = [r for r in results if r['target'] == 95 and r['subset'] == 'all']
    save_csv('primary95_all_comparisons.csv', primary)
    provenance = {'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [SOURCE95 / 'outer_fold_metrics.csv', SOURCE90 / 'outer_fold_metrics.csv']},
                  'scipy_version': scipy.__version__, 'classroom_example1': example1,
                  'classroom_example2': example2, 'calibration_files_verified': len(calibration_files),
                  'fold_sizes': {'fitting': 1380, 'calibration': 348, 'test': 192},
                  'comparisons': len(results), 'course_variance_divisor': 10,
                  'primary_family_size_per_metric': 45, 'multiple_comparison_adjustment': None,
                  'status': 'post-hoc exploratory classroom approximation; dependent CV folds',
                  'no_retraining': True, 'no_training_code_changes': True}
    (OUT / 'verification.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Verification passed. Primary: original 95% target, all negatives.')
    for r in primary:
        if r['local'] == 'F4.2':
            print(f"{r['algorithm']:6s} {r['condition']:12s} {r['metric']:7s} "
                  f"delta={r['local_minus_baseline']:+.6f} p={r['p']:.6f} "
                  f"x={r['statistic']:+.4f} df={r['df']} reject={r['reject']}")
    for metric in ['FPR', 'TPR', 'ROC_AUC']:
        selected = [r for r in primary if r['metric'] == metric]
        print(metric, 'rejections', sum(r['reject'] for r in selected), '/', len(selected))


if __name__ == '__main__':
    main()
