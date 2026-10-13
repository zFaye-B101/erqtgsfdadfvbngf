"""Audit returned local-feature and recall-90 ZIPs; publish model-free evidence.

Uses only the standard library. Never imports experiment code or loads models.
Both output directories must be new. Source ZIPs remain read-only.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
from collections import defaultdict
from zipfile import ZipFile

METRICS = ('accuracy', 'precision', 'TPR', 'FPR', 'F1',
           'balanced_accuracy', 'ROC_AUC', 'average_precision')
KEYS = ('fold', 'algorithm', 'features', 'condition', 'negative_subset')
PAIR = ('query_image_id', 'reference_image_id', 'query_product_id',
        'reference_product_id', 'label', 'kind')
SOURCES = ('sop_baselines.py', 'sop_supcon.py', 'sop_reporting.py', 'sop_config.py',
           'sop_feature_bank.py', 'sop_features.py', 'sop_protocol.py',
           'sop_experiment.py', 'sop_calibration.py', 'sop_tuning.py')
LOCAL = 'ResNet_LocalFeature_Comparison'
JSON_NAMES = {'protocol.json', 'calibration.json', 'tuning.json', 'completed.json',
              'all_tuning_records.json', 'grid_requirement_audit.json',
              'paper_adaptation.json', 'skipped_incompatible_comparisons.json',
              'recalibration_info.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    require(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10),
            f'Numeric mismatch: {a} != {b}')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def csv_rows(z, name):
    data = z.read(name)
    if name.endswith('.gz'):
        data = gzip.decompress(data)
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def metrics(rows, threshold):
    tp = fp = tn = fn = 0
    scores = defaultdict(lambda: [0, 0])
    for row in rows:
        y, score = int(row['label']), float(row['score'])
        require(y in (0, 1) and math.isfinite(score), 'Invalid label/score')
        prediction = int(score >= threshold)
        require(prediction == int(row['prediction']), 'Prediction mismatch')
        close(row['threshold'], threshold)
        tp += y * prediction
        fp += (1-y) * prediction
        tn += (1-y) * (1-prediction)
        fn += y * (1-prediction)
        scores[score][y] += 1
    positive, negative = tp + fn, fp + tn
    require(positive > 0 and negative > 0, 'Both labels required')
    seen_negative = wins = 0
    for score, (n, p) in sorted(scores.items()):
        wins += p * (seen_negative + n / 2)
        seen_negative += n
    seen_positive = seen = ap = 0
    for score, (n, p) in sorted(scores.items(), reverse=True):
        seen_positive += p
        seen += n + p
        ap += p / positive * seen_positive / seen
    tpr, fpr = tp / positive, fp / negative
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, pair_count=len(rows),
                accuracy=(tp+tn)/len(rows), precision=tp/(tp+fp) if tp+fp else 0,
                TPR=tpr, FPR=fpr, F1=2*tp/(2*tp+fp+fn),
                balanced_accuracy=(tpr+1-fpr)/2, ROC_AUC=wins/(positive*negative),
                average_precision=ap)


def validate_names(z):
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    require(len(names) == len(set(names)), 'Duplicate ZIP entries')
    for name in names:
        p = PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and
                '\\' not in name and ':' not in name, 'Unsafe ZIP entry')
    return names


def audit_run(z, root, run, fingerprints, original=None):
    prefix = f'{root}/{run}/'
    protocols = sorted(n for n in z.namelist() if n.startswith(prefix)
                       and n.endswith('/protocol.json'))
    require(len(protocols) == (8 if run == LOCAL else 10), 'Missing groups/families')
    table, summaries, pairs, identities = {}, [], {}, []
    counts = dict(predictions=0, calibration_models=0, completed_records=0,
                  metric_rows=0, summary_rows=0, paired_delta_rows=0)
    for name in protocols:
        identity = json.loads(z.read(name))
        cfg = identity['config']
        group, family = name.split('/')[2:4]
        base = name.rsplit('/', 1)[0] + '/'
        expected_target = .9 if original else .95
        require(cfg['target_validation_recall'] == expected_target and
                cfg['outer_folds'] == 10 and cfg['inner_folds'] == 3 and
                cfg['mode'] == 'formal' and not cfg['run_official_test'], 'Protocol mismatch')
        require(identity['code_fingerprint'] == fingerprints[run], 'Source fingerprint mismatch')
        identities.append(identity)
        rows = csv_rows(z, base + 'outer_fold_metrics.csv')
        algorithms = ['LR', 'SVM'] if family == 'baselines' else ['SupCon']
        lookup = {tuple(str(r[k]) for k in KEYS): r for r in rows}
        require(len(lookup) == len(rows) == len(algorithms)*len(cfg['feature_sets'])*150,
                'Duplicate or missing metric rows')
        for fold in range(1, 11):
            if not original:
                completed = json.loads(z.read(base+f'outer_{fold:02d}/completed.json'))['rows']
                require(len(completed) == len(algorithms)*len(cfg['feature_sets'])*15, 'Completion count')
                for r in completed:
                    actual = lookup[tuple(str(r[k]) for k in KEYS)]
                    for m in METRICS:
                        close(r[m], actual[m])
                counts['completed_records'] += 1
            for alg in algorithms:
                for feature in cfg['feature_sets']:
                    directory = base+f'outer_{fold:02d}/{alg}_{feature}/'
                    cal = json.loads(z.read(directory+'calibration.json'))
                    calibration = csv_rows(z, directory+'calibration_predictions.csv.gz')
                    positive_scores = sorted((float(r['score']) for r in calibration
                                              if r['label'] == '1'), reverse=True)
                    threshold = positive_scores[math.ceil(expected_target*len(positive_scores))-1]
                    close(cal['threshold'], threshold)
                    source_dir = directory.replace(root+'/', 'results/', 1)
                    source_cal = json.loads(original.read(source_dir+'calibration.json')) if original else cal
                    fit, cal_products = set(source_cal['fitting_products']), set(source_cal['calibration_products'])
                    require(fit and cal_products and not fit & cal_products, 'Fit/calibration overlap')
                    observed_cal = {int(r[k]) for r in calibration for k in ('query_product_id', 'reference_product_id')}
                    require(observed_cal <= cal_products, 'Calibration product mismatch')
                    if original:
                        old_cal = csv_rows(original, source_dir+'calibration_predictions.csv.gz')
                        require(len(old_cal) == len(calibration), 'Calibration length mismatch')
                        for old, new in zip(old_cal, calibration):
                            require(all(old[k] == new[k] for k in PAIR), 'Calibration pairs changed')
                            close(old['score'], new['score'])
                    counts['calibration_models'] += 1
                    for condition in cfg['conditions']:
                        filename = 'predictions_'+condition+'.csv.gz'
                        pred = csv_rows(z, directory+filename)
                        pair_hash = digest(json.dumps([[r[k] for k in PAIR] for r in pred]).encode())
                        require(pairs.setdefault(fold, pair_hash) == pair_hash, 'Test pairs differ')
                        products = {int(r[k]) for r in pred for k in ('query_product_id', 'reference_product_id')}
                        require(not products & (fit | cal_products), 'Test leakage')
                        require(all((r['query_product_id'] == r['reference_product_id']) == (r['label'] == '1')
                                    for r in pred), 'Labels/product IDs disagree')
                        if original:
                            old_pred = csv_rows(original, source_dir+filename)
                            require(len(old_pred) == len(pred), 'Prediction length changed')
                            for old, new in zip(old_pred, pred):
                                require(all(old[k] == new[k] for k in PAIR), 'Recalibrated pairs changed')
                                close(old['score'], new['score'])
                                close(old['threshold'], new['original_threshold_95'])
                        for subset in ('all', 'random', 'colour_hard'):
                            selected = pred if subset == 'all' else [r for r in pred if r['kind'] in ('positive', subset)]
                            actual = lookup[(str(fold), alg, feature, condition, subset)]
                            close(actual['threshold'], threshold)
                            for m, value in metrics(selected, threshold).items():
                                close(actual[m], value)
                        counts['predictions'] += 1
        for r in rows:
            table[(group,)+tuple(str(r[k]) for k in KEYS)] = r
        counts['metric_rows'] += len(rows)
        for r in csv_rows(z, base+'summary_mean_sd.csv'):
            values = [float(v[r['metric']]) for v in rows
                      if all(v[k] == r[k] for k in KEYS[1:])]
            require(len(values) == 10, 'Summary fold count')
            close(r['mean'], statistics.mean(values))
            close(r['sd_across_outer_folds'], statistics.stdev(values))
            close(r['outer_fold_count'], 10)
            summaries.append(dict(r, group=group, run=run))
        if original:
            old_rows = {tuple(str(r[k]) for k in KEYS): r for r in
                        csv_rows(original, base.replace(root+'/', 'results/', 1)+'outer_fold_metrics.csv')}
            comparison = csv_rows(z, base+'comparison_95_vs_90.csv')
            require(len(comparison) == len(rows), '95-vs-90 count')
            for r in comparison:
                key = tuple(str(r[k]) for k in KEYS)
                for m in METRICS + ('threshold',):
                    close(r[m+'_95'], old_rows[key][m])
                    close(r[m+'_90'], lookup[key][m])
                for m in ('FPR', 'TPR', 'accuracy', 'precision', 'F1', 'balanced_accuracy'):
                    close(r[m+'_delta_90_minus_95'], float(lookup[key][m])-float(old_rows[key][m]))
    require(len({d['shared_protocol_id'] for d in identities}) == 1, 'Protocol IDs differ')
    if not original:
        deltas = csv_rows(z, prefix+'comparisons/paired_group_deltas.csv')
        for r in deltas:
            key = tuple(r[k] for k in ('fold', 'algorithm'))
            tail = tuple(r[k] for k in ('condition', 'negative_subset'))
            a = table[(r['added_group'],)+key+(r['added_features'],)+tail]
            b = table[(r['baseline_group'],)+key+(r['baseline_features'],)+tail]
            close(r['delta_added_minus_baseline'], float(a[r['metric']])-float(b[r['metric']]))
        require(not json.loads(z.read(prefix+'comparisons/skipped_incompatible_comparisons.json')),
                'Skipped comparisons')
        counts['paired_delta_rows'] = len(deltas)
    counts.update(run=run, code_fingerprint=fingerprints[run],
                  shared_protocol_id=identities[0]['shared_protocol_id'],
                  summary_rows=len(summaries), test_pair_hashes=pairs)
    return counts, table, summaries


def portable(value, key=''):
    if isinstance(value, dict):
        return {k: portable(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, key) for v in value]
    if isinstance(value, str) and (re.match(r'^[A-Za-z]:[\\/]', value) or value.startswith(('/Users/', '/home/'))):
        return value.replace('\\', '/').rstrip('/').split('/')[-1]
    return value


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export(z, root, runs, output, audit, tables, summaries):
    output.mkdir(parents=True, exist_ok=False)
    manifest, excluded = [], []
    for name in sorted(validate_names(z)):
        rel = name[len(root)+1:]
        require(name.startswith(root+'/'), 'Unexpected ZIP root')
        if not any(rel.startswith(run+'/') for run in runs) and rel != 'recalibration_info.json':
            continue
        path = PurePosixPath(rel)
        allowed = (path.name in JSON_NAMES or path.suffix == '.csv' or
                   path.name.startswith(('predictions_', 'calibration_predictions')) and path.name.endswith('.csv.gz') or
                   path.suffix == '.png' and 'figures' in path.parts)
        if not allowed:
            excluded.append(rel)
            continue
        data = z.read(name)
        original_hash = digest(data)
        if path.suffix == '.json':
            original = json.loads(data)
            cleaned = portable(original)
            if original != cleaned:
                data = (json.dumps(cleaned, indent=2)+'\n').encode()
        target = output/'artifacts'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest.append(dict(path='artifacts/'+rel, source_sha256=original_hash,
                             published_sha256=digest(data), path_redacted=digest(data)!=original_hash))
    write_csv(output/'all_metrics.csv', summaries)
    write_csv(output/'outer_fold_metrics.csv', [dict(r, group=k[0], run=run)
              for run, table in tables.items() for k, r in table.items()])
    write_csv(output/'key_metrics.csv', [r for r in summaries if r['condition'] in ('clean', 'occlusion25')
              and r['negative_subset'] in ('all', 'random') and r['metric'] in ('ROC_AUC', 'TPR', 'FPR')])
    decisions = []
    for run, table in tables.items():
        if run != LOCAL:
            continue
        groups = sorted({k[0] for k in table})
        baseline = next(g for g in groups if g.startswith('F3_'))
        for group in groups:
            if group == baseline:
                continue
            for alg in ('LR', 'SVM', 'SupCon'):
                for cond in ('clean', 'occlusion25', 'noise005', 'blur15', 'rotation15'):
                    for subset in ('all', 'random', 'colour_hard'):
                        a = sorted([(k,r) for k,r in table.items() if k[0]==group and k[2]==alg and k[4:]==(cond,subset)], key=lambda x:int(x[0][1]))
                        b = sorted([(k,r) for k,r in table.items() if k[0]==baseline and k[2]==alg and k[4:]==(cond,subset)], key=lambda x:int(x[0][1]))
                        require(len(a)==len(b)==10, 'Decision fold count')
                        d = {m:[float(x[1][m])-float(y[1][m]) for x,y in zip(a,b)] for m in ('TPR','FPR','ROC_AUC')}
                        count = sum(v < 0 for v in d['FPR'])
                        row = dict(added_group=group, baseline_group=baseline, algorithm=alg,
                                   condition=cond, negative_subset=subset, fpr_improved_folds=count)
                        for m,v in d.items():
                            row[m+'_delta_mean']=statistics.mean(v)
                            row[m+'_delta_sd']=statistics.stdev(v)
                        row['meets_descriptive_rule']=count>=8 and statistics.mean(d['FPR'])<0 and statistics.mean(d['TPR'])>=-.01
                        decisions.append(row)
    write_csv(output/'local_feature_decisions.csv', decisions)
    audit.update(published_files=len(manifest), excluded_files=excluded,
                 excluded_note='Weights, training checkpoints/loss traces, caches and source ZIP are not exported.')
    write_json(output/'file_manifest.json', manifest)
    write_json(output/'audit.json', audit)
    for r in manifest:
        require(digest((output/r['path']).read_bytes())==r['published_sha256'], 'Export hash mismatch')
    print(output.name, 'published',len(manifest),'excluded',len(excluded),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--recall90',type=Path,required=True)
    p.add_argument('--output-local',type=Path,required=True)
    p.add_argument('--output-recall90',type=Path,required=True)
    args=p.parse_args()
    require(not args.output_local.exists() and not args.output_recall90.exists(), 'Output already exists')
    require(args.output_local.resolve()!=args.output_recall90.resolve(), 'Outputs must differ')
    repo=Path(__file__).resolve().parents[1]
    current=b''.join((repo/'PythonCode/utils'/n).read_bytes() for n in SOURCES)
    old=b''.join(subprocess.check_output(['git','show','c275cd4:PythonCode/utils/'+n],cwd=repo) for n in SOURCES)
    fingerprints={LOCAL:digest(current),'Main_Experiments':digest(old)}
    limits=['No models loaded or retraining performed.', 'Full preprocessing and all inner-fold memberships not supplied.',
            'Official holdout unused; descriptive comparisons are not significance tests.',
            '90% is a supplementary operating-point analysis, not a replacement for the original 95% criterion.']
    with ZipFile(args.source) as z, ZipFile(args.recall90) as w:
        validate_names(z);validate_names(w)
        info=json.loads(w.read('results_recall90/recalibration_info.json'))
        require(info['script_sha256']==digest((repo/'PythonCode/03_SOP_Recalibrate_90.py').read_bytes()), 'Recalibration script fingerprint')
        records95=[];records90=[];tables95={};tables90={};summaries95=[];summaries90=[]
        for run in ('Main_Experiments',LOCAL):
            rec,table,summary=audit_run(z,'results',run,fingerprints)
            records95.append(rec)
            if run==LOCAL:tables95[run]=table;summaries95.extend(summary)
            print('95% verified',run,rec['predictions'],flush=True)
            rec,table,summary=audit_run(w,'results_recall90',run,fingerprints,original=z)
            records90.append(rec);tables90[run]=table;summaries90.extend(summary)
            print('90% verified',run,rec['predictions'],flush=True)
        common=dict(date='2026-10-13',limits=limits,all_eight_metrics_recomputed=True,
                    summary_mean_sample_sd_verified=True,fit_calibration_test_disjoint=True)
        export(z,'results',[LOCAL],args.output_local,
               dict(common,round='R2',source='results.zip',runs=[records95[1]],
                    source_fingerprint_reference='PythonCode/utils at f412219'),tables95,summaries95)
        export(w,'results_recall90',['Main_Experiments',LOCAL],args.output_recall90,
               dict(common,round='R1/R2 postprocessing, no new training round',source='results_recall90.zip',
                    runs=records90,source_95_audits=records95,scores_and_pairs_unchanged=True,
                    thresholds_recomputed_from_calibration_positives=True,recalibration_script_sha256=info['script_sha256']),
               tables90,summaries90)


if __name__=='__main__':
    main()
