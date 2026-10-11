"""Calibrate a fixed final model on held-out training products only."""
import numpy as np
from sop_protocol import manual_metrics, recall_threshold, write_json


def split_calibration_products(frame, products, fraction, inner_folds, seed):
    """Reserve >=2 products per category and retain enough for inner pairing."""
    table = frame[frame.class_id.isin(products)].groupby('class_id').super_class_id.first()
    rng = np.random.default_rng(seed)
    fitting, calibration = [], []
    for category, group in table.groupby(table):
        minimum_fit = 2 * inner_folds
        if len(group) < minimum_fit + 2:
            raise ValueError(f'Category {category} needs >= {minimum_fit + 2} training products for independent calibration.')
        count = max(2, min(int(np.ceil(len(group) * fraction)), len(group) - minimum_fit))
        shuffled = rng.permutation(group.index.to_numpy())
        calibration.extend(int(p) for p in shuffled[:count])
        fitting.extend(int(p) for p in shuffled[count:])
    return sorted(fitting), sorted(calibration)


def calibrate(context, scores, config, directory):
    """Do not refit the model after this threshold has been chosen."""
    pairs = context['calibration_pairs']
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise ValueError('Calibration scores must be finite and match the pair count.')
    threshold = recall_threshold(pairs.label.to_numpy(), scores, config['target_validation_recall'])
    record = {'threshold': threshold, 'target_recall': config['target_validation_recall'],
              'calibration_products': context['calibration_products'],
              'fitting_products': context['train_products'],
              'pair_count': len(pairs), 'positive_count': int(pairs.label.sum()),
              'metrics': manual_metrics(pairs.label.to_numpy(), scores, threshold),
              'note': 'Fixed final model; disjoint calibration products; no refit after calibration. Test recall is not guaranteed.'}
    write_json(directory / 'calibration.json', record)
    predictions = pairs.copy()
    predictions['score'] = scores
    predictions['threshold'] = threshold
    predictions.to_csv(directory / 'calibration_predictions.csv.gz', index=False)
    return threshold
