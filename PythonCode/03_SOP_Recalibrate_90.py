"""Recalibrate saved 95%-recall experiments to 90% without fitting any model.

Run after 02 has saved calibration and test predictions, or read results.zip.
Inputs are read-only. A new output directory is required; existing output is
never overwritten. Thresholds use clean calibration labels only, not test labels.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd

from utils.sop_protocol import manual_metrics, recall_threshold, write_json
from utils.sop_reporting import report_results

ROOT = Path(__file__).resolve().parent


class SavedResults:
    """Read a results directory or ZIP without extracting or loading models."""
    def __init__(self, source):
        self.source = Path(source).expanduser().resolve()
        self.archive = None
        if self.source.is_dir():
            self.names = {p.relative_to(self.source).as_posix() for p in self.source.rglob('*') if p.is_file()}
        elif self.source.is_file() and zipfile.is_zipfile(self.source):
            self.archive = zipfile.ZipFile(self.source)
            entries = [n for n in self.archive.namelist() if not n.endswith('/')]
            if len(entries) != len(set(entries)):
                raise ValueError('ZIP contains duplicate filenames.')
            self.names = set(entries)
        else:
            raise FileNotFoundError('Select a completed results directory or ZIP: ' + str(self.source))
        for name in self.names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe saved result path: ' + name)

    def read(self, name):
        return self.archive.read(name) if self.archive else (self.source / name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def csv(self, name):
        return pd.read_csv(io.BytesIO(self.read(name)), compression='gzip' if name.endswith('.gz') else None)

    def close(self):
        if self.archive:
            self.archive.close()


def reprocess(source, output):
    """Create fresh metrics using each final model's independent calibration set."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output == source or (source.is_dir() and (output.is_relative_to(source) or source.is_relative_to(output))):
        raise ValueError('Output must be separate from the original results directory.')
    if output.exists():
        raise FileExistsError('Output already exists; nothing was overwritten. Choose another --output: ' + str(output))
    saved = SavedResults(source)
    try:
        jobs = []
        for name in sorted(saved.names):
            if not name.endswith('/outer_fold_metrics.csv'):
                continue
            directory = PurePosixPath(name).parent
            identity = saved.json(str(directory / 'protocol.json'))
            config = identity['config']
            if config.get('target_validation_recall') != .95:
                raise ValueError('Expected original 95%-recall configuration: ' + str(directory))
            table = saved.csv(name)
            required = {'fold', 'algorithm', 'features', 'condition', 'negative_subset', 'FPR', 'TPR'}
            if not required <= set(table.columns) or table.empty:
                raise ValueError('Incomplete saved metric table: ' + name)
            if table.duplicated(['fold', 'algorithm', 'features', 'condition', 'negative_subset']).any():
                raise ValueError('Duplicate saved metric rows: ' + name)
            if not table.groupby(['algorithm', 'features', 'condition', 'negative_subset']).fold.nunique().eq(config['outer_folds']).all():
                raise ValueError('All outer folds must be complete: ' + name)
            for (fold, algorithm, feature), rows in table.groupby(['fold', 'algorithm', 'features']):
                model = directory / f'outer_{int(fold):02d}' / (algorithm + '_' + feature)
                needed = [str(model / 'calibration_predictions.csv.gz')] + [str(model / ('predictions_' + c + '.csv.gz')) for c in config['conditions']]
                if not set(needed) <= saved.names:
                    raise ValueError('Saved calibration/test predictions are missing: ' + str(model))
            relative = directory.relative_to('results') if directory.parts[0] == 'results' else directory
            jobs.append((directory, relative, identity, table))
        if not jobs:
            raise ValueError('No completed experiment tables found.')
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'recalibration_info.json', {
            'target_recall': .90, 'original_target_recall': .95, 'source': str(source),
            'selection': 'Clean held-out calibration positives only; no test threshold tuning.',
            'training': 'No feature extraction, model fitting or hyperparameter search.',
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'note': '90% calibration recall does not guarantee 90% test recall or FPR below 50%.'})
        family_tables = {}
        for directory, relative, original_identity, old_table in jobs:
            destination = output / str(relative)
            config = dict(original_identity['config'])
            config.update(target_validation_recall=.90, results_dir=str(output))
            records = []
            for (fold, algorithm, feature), old_rows in old_table.groupby(['fold', 'algorithm', 'features']):
                model = directory / f'outer_{int(fold):02d}' / (algorithm + '_' + feature)
                target = destination / f'outer_{int(fold):02d}' / (algorithm + '_' + feature)
                target.mkdir(parents=True, exist_ok=True)
                cal = saved.csv(str(model / 'calibration_predictions.csv.gz'))
                threshold = recall_threshold(cal.label.to_numpy(), cal.score.to_numpy(), .90)
                write_json(target / 'calibration.json', {'threshold': threshold, 'target_recall': .90,
                           'metrics': manual_metrics(cal.label.to_numpy(), cal.score.to_numpy(), threshold),
                           'note': 'Original final model reused. No test labels used for threshold selection.'})
                cal['threshold'] = threshold
                cal.to_csv(target / 'calibration_predictions.csv.gz', index=False)
                for condition in config['conditions']:
                    pred = saved.csv(str(model / ('predictions_' + condition + '.csv.gz')))
                    pred['original_threshold_95'] = pred['threshold']
                    pred['threshold'] = threshold
                    pred['prediction'] = (pred.score >= threshold).astype(int)
                    pred.to_csv(target / ('predictions_' + condition + '.csv.gz'), index=False)
                    for subset in ('all', 'random', 'colour_hard'):
                        selected = pred if subset == 'all' else pred[pred.kind.isin(['positive', subset])]
                        row = dict(fold=int(fold), algorithm=algorithm, features=feature, condition=condition,
                                   negative_subset=subset, threshold=threshold, pair_count=len(selected))
                        row.update(manual_metrics(selected.label.to_numpy(), selected.score.to_numpy(), threshold))
                        records.append(row)
            table = pd.DataFrame(records)
            table.to_csv(destination / 'outer_fold_metrics.csv', index=False)
            report_results(table, [], destination, config)
            identity = {**original_identity, 'config': config, 'recalibration_target': .90,
                        'original_protocol': original_identity,
                        'note': 'Postprocessing only; original model/tuning files remain at the source.'}
            write_json(destination / 'protocol.json', identity)
            comparison = old_table.merge(table, on=['fold','algorithm','features','condition','negative_subset'],
                                         suffixes=('_95','_90'), validate='one_to_one')
            for metric in ('FPR','TPR','accuracy','precision','F1','balanced_accuracy'):
                comparison[metric + '_delta_90_minus_95'] = comparison[metric+'_90'] - comparison[metric+'_95']
            comparison.to_csv(destination / 'comparison_95_vs_90.csv', index=False)
            family_tables.setdefault(destination.parent, []).append(pd.read_csv(destination / 'summary_mean_sd.csv'))
            print('Saved 90% recalibration:', destination, flush=True)
        for group, tables in family_tables.items():
            pd.concat(tables, ignore_index=True).to_csv(group / 'combined_summary_mean_sd.csv', index=False)
        print('Original results preserved. New results:', output)
    finally:
        saved.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Original results directory or results.zip; read-only.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results_recall90')
    args = parser.parse_args(argv)
    source = args.source
    if source is None:
        source = ROOT / 'results.zip' if (ROOT / 'results.zip').is_file() else ROOT / 'results'
    reprocess(source, args.output)


if __name__ == '__main__':
    main()
