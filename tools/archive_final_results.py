"""Publish the audited final F1.1/F2.1 snapshot without loading models.

Uses the existing independent standard-library auditor. Original 2026-10-13
archive behaviour remains strict; mixed fingerprints are allowed only for the
explicitly listed final Main_Experiments groups.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
from zipfile import ZipFile

from archive_local_results import (LOCAL, SOURCES, METRICS, audit_run, digest,
                                 export, require, validate_names, write_csv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--recall90', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.name == 'nt':
        args.output = Path('\\\\?\\' + str(args.output.resolve()))
    require(not args.output.exists(), 'Output already exists')
    repo = Path(__file__).resolve().parents[1]
    current = digest(b''.join((repo/'PythonCode/utils'/n).read_bytes() for n in SOURCES))
    old = digest(b''.join(subprocess.check_output(['git','show','c275cd4:PythonCode/utils/'+n],cwd=repo) for n in SOURCES))
    fingerprints = {'Main_Experiments': old, LOCAL: current}
    for group in ['F1.1_Traditional_Content_HSV48','F2.1_Traditional_Content_HSV_Gradient192']:
        fingerprints[('Main_Experiments',group)] = current
    with ZipFile(args.source) as source, ZipFile(args.recall90) as recall:
        for zipped in [source, recall]:
            names = validate_names(zipped)
            require(not any('/F5_' in n or '/F1.2_' in n for n in names), 'Stale group in final snapshot')
        info = json.loads(recall.read('results_recall90/recalibration_info.json'))
        script_bytes = (repo/'PythonCode/03_SOP_Recalibrate_90.py').read_bytes()
        script_lf = script_bytes.replace(b'\r\n', b'\n')
        script_hashes = {digest(script_bytes), digest(script_lf), digest(script_lf.replace(b'\n', b'\r\n'))}
        require(info['script_sha256'] in script_hashes, '03 fingerprint (including Git newline variants)')
        audits = [[],[]]; tables = [{},{}]; summaries = [[],[]]
        for run in ['Main_Experiments',LOCAL]:
            for index,zipped,root in [(0,source,'results'),(1,recall,'results_recall90')]:
                record,table,summary = audit_run(zipped,root,run,fingerprints,
                    original=source if index else None,
                    expected_families=12 if run=='Main_Experiments' else 8,
                    allow_skipped=run=='Main_Experiments')
                audits[index].append(record);tables[index][run]=table;summaries[index].extend(summary)
                print('Verified',run,root,record['predictions'],flush=True)
        common = dict(date='2026-10-14', all_eight_metrics_recomputed=True,
            summary_mean_sample_sd_verified=True, fit_calibration_test_disjoint=True,
            limits=['No model loading or retraining; no full preprocessing/inner-membership audit.',
                    'Official holdout unused; fold SD is not a confidence interval.',
                    'Mixed-version comparisons are explicit descriptive postprocessing, not automatic compatible-run comparisons.'])
        for index,zipped,root,target in [(0,source,'results','recall95'),(1,recall,'results_recall90','recall90')]:
            export(zipped,root,['Main_Experiments',LOCAL],args.output/target,
                   dict(common,source=args.source.name if not index else args.recall90.name,
                        runs=audits[index],threshold_target=.95 if not index else .9,
                        scores_and_pairs_unchanged=bool(index)),tables[index],summaries[index])
        comparisons = []
        for index in [0,1]:
            table=tables[index]['Main_Experiments']
            for base,added in [('F1_Traditional_HSV48','F1.1_Traditional_Content_HSV48'),
                               ('F2_Traditional_HSV_Gradient192','F2.1_Traditional_Content_HSV_Gradient192')]:
                for alg in ['LR','SVM','SupCon']:
                    for cond in ['clean','occlusion25','noise005','blur15','rotation15']:
                        for subset in ['all','random','colour_hard']:
                            a=sorted([(k,r) for k,r in table.items() if k[0]==added and k[2]==alg and k[4:]==(cond,subset)],key=lambda x:int(x[0][1]))
                            b=sorted([(k,r) for k,r in table.items() if k[0]==base and k[2]==alg and k[4:]==(cond,subset)],key=lambda x:int(x[0][1]))
                            require(len(a)==len(b)==10,'Padding comparison fold count')
                            for metric in METRICS:
                                delta=[float(x[1][metric])-float(y[1][metric]) for x,y in zip(a,b)]
                                comparisons.append(dict(target_recall=.95 if not index else .9,
                                    baseline_group=base,added_group=added,algorithm=alg,condition=cond,
                                    negative_subset=subset,metric=metric,mean_delta=statistics.mean(delta),
                                    sd_delta=statistics.stdev(delta),negative_delta_folds=sum(v<0 for v in delta),
                                    comparison_scope='mixed-source versions; audited pairing and reviewed relevant implementations'))
        write_csv(args.output/'padding_comparisons.csv',comparisons)


if __name__=='__main__':
    main()
