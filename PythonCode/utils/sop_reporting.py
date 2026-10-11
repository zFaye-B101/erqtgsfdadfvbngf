"""Manual metric reporting shared by all matching algorithms."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sop_protocol import manual_metrics, write_json


def evaluate_scores(pairs, score_by_condition, threshold, output_dir, fold, algorithm, feature_name):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    y = pairs.label.to_numpy()
    kinds = pairs.kind.to_numpy()
    for condition, score in score_by_condition.items():
        score = np.asarray(score, dtype=float)
        if score.shape != y.shape or not np.isfinite(score).all():
            raise ValueError("Invalid matching scores.")
        predictions = pairs.copy()
        predictions["condition"] = condition
        predictions["score"] = score
        predictions["threshold"] = threshold
        predictions["prediction"] = (score >= threshold).astype(int)
        predictions.to_csv(directory / f"predictions_{condition}.csv.gz", index=False)
        for subset in ("all", "random", "colour_hard"):
            mask = np.ones(len(y), dtype=bool) if subset == "all" else ((kinds == "positive") | (kinds == subset))
            row = {"fold": fold, "algorithm": algorithm, "features": feature_name,
                   "condition": condition, "negative_subset": subset, "threshold": float(threshold),
                   "pair_count": int(mask.sum())}
            row.update(manual_metrics(y[mask], score[mask], threshold))
            rows.append(row)
    return rows


def report_results(results, tuning, directory, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "precision", "TPR", "FPR", "F1", "balanced_accuracy", "ROC_AUC", "average_precision"]
    keys = ["algorithm", "features", "condition", "negative_subset"]
    rows = []
    for key, group in results.groupby(keys, sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            rows.append({**dict(zip(keys, key)), "metric": metric, "mean": float(values.mean()),
                         "sd_across_outer_folds": float(values.std(ddof=1)), "outer_fold_count": len(values),
                         "mode": config["mode"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(directory / "summary_mean_sd.csv", index=False)
    deltas = []
    for (algorithm, condition, subset), group in results.groupby(["algorithm", "condition", "negative_subset"]):
        for metric in ("FPR", "TPR", "F1", "ROC_AUC"):
            pivot = group.pivot(index="fold", columns="features", values=metric)
            for baseline, added in config["comparisons"]:
                for fold, delta in (pivot[added] - pivot[baseline]).items():
                    deltas.append({"algorithm": algorithm, "condition": condition, "negative_subset": subset,
                                   "metric": metric, "baseline_features": baseline, "added_features": added,
                                   "fold": int(fold), "delta_added_minus_baseline": float(delta)})
    columns = ["algorithm", "condition", "negative_subset", "metric", "baseline_features", "added_features", "fold", "delta_added_minus_baseline"]
    delta_frame = pd.DataFrame(deltas, columns=columns)
    delta_frame.to_csv(directory / "paired_feature_deltas.csv", index=False)
    delta_rows = []
    for key, group in delta_frame.groupby(columns[:6]):
        values = group.delta_added_minus_baseline.to_numpy(dtype=float)
        delta_rows.append({**dict(zip(columns[:6], key)), "mean_delta": float(values.mean()),
                           "sd_across_outer_folds": float(values.std(ddof=1))})
    pd.DataFrame(delta_rows, columns=columns[:6] + ["mean_delta", "sd_across_outer_folds"]).to_csv(
        directory / "paired_delta_mean_sd.csv", index=False)
    figure_dir = directory / "figures"
    figure_dir.mkdir(exist_ok=True)
    conditions = list(config["conditions"])
    for subset in ("random", "colour_hard"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for axis, metric in zip(axes.ravel(), ("FPR", "TPR", "F1", "ROC_AUC")):
            for algorithm in results.algorithm.unique():
                for name in config["feature_sets"]:
                    part = summary[(summary.algorithm == algorithm) & (summary.features == name)
                                   & (summary.negative_subset == subset) & (summary.metric == metric)]
                    part = part.set_index("condition").loc[conditions]
                    axis.errorbar(np.arange(len(conditions)), part["mean"], yerr=part.sd_across_outer_folds,
                                  marker="o", capsize=3, label=algorithm + " / " + name)
            axis.set_xticks(np.arange(len(conditions)), conditions, rotation=20)
            axis.set_title(metric + " (mean +/- 1 outer-fold SD)")
            axis.set_ylim(-0.05, 1.05)
            axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(config["mode"].upper() + " / " + subset + " negatives")
        fig.savefig(figure_dir / (subset + "_robustness.png"), dpi=160)
        plt.close(fig)
    audit = []
    if tuning:
        for (algorithm, name), group in pd.DataFrame(tuning).groupby(["algorithm", "features"]):
            records = group.to_dict("records")
            unresolved = sum(bool(record["boundary_unresolved"]) for record in records)
            audit.append({"algorithm": algorithm, "features": name,
                          "boundary_selected_folds": unresolved, "middle_selected_folds": len(records) - unresolved,
                          "middle_is_majority": len(records) - unresolved > len(records) / 2})
    write_json(directory / "grid_requirement_audit.json", audit)
    for row in audit:
        if not row["middle_is_majority"]:
            print("ATTENTION: Hyperparameter range needs review using inner validation only:", row)


def print_tables(directory):
    directory = Path(directory)
    table = pd.read_csv(directory / "summary_mean_sd.csv")
    print("\nOuter-fold summary:")
    print(table.loc[table.metric.isin(["FPR", "TPR", "F1", "ROC_AUC"])].to_string(index=False))
    print("Results:", directory)


def combine_results(experiment):
    base = Path(experiment.config["results_dir"]) / experiment.config["run_name"] / experiment.config["feature_profile"]
    summaries = []
    fingerprints = []
    for family in ("baselines", "supcon"):
        directory = base / family
        summary_path = directory / "summary_mean_sd.csv"
        if not summary_path.exists():
            return
        identity = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        if identity["shared_protocol_id"] != experiment.protocol_id or identity["config"] != experiment.config:
            raise ValueError("Cannot combine runs with different protocol/configuration.")
        table = pd.read_csv(summary_path)
        if not table.outer_fold_count.eq(experiment.config["outer_folds"]).all():
            raise ValueError("Cannot combine incomplete outer-fold summaries.")
        summaries.append(table)
        fingerprints.append(identity["code_fingerprint"])
    if len(set(fingerprints)) != 1:
        raise ValueError("Cannot combine results produced by different code versions.")
    pd.concat(summaries, ignore_index=True).to_csv(base / "combined_summary_mean_sd.csv", index=False)
    print("Combined LR/SVM/SupCon summary:", base / "combined_summary_mean_sd.csv")


def update_comparisons(run_directory):
    """Compare completed groups only when protocol, code and common settings agree."""
    from itertools import combinations
    root = Path(run_directory)
    output = root / 'comparisons'
    output.mkdir(parents=True, exist_ok=True)
    records, summaries, skipped = [], [], []
    for family in ('baselines', 'supcon'):
        runs = []
        for path in sorted(root.glob('*/' + family + '/outer_fold_metrics.csv')):
            identity_path = path.parent / 'protocol.json'
            if not identity_path.exists():
                continue
            identity = json.loads(identity_path.read_text(encoding='utf-8'))
            table = pd.read_csv(path)
            count = identity['config']['outer_folds']
            if not table.groupby(['algorithm', 'features', 'condition', 'negative_subset']).fold.nunique().eq(count).all():
                continue
            group = path.parent.parent.name
            for key, frame in table.groupby(['algorithm', 'features', 'condition', 'negative_subset']):
                for metric in ('FPR', 'TPR', 'F1', 'ROC_AUC'):
                    summaries.append(dict(zip(['algorithm', 'features', 'condition', 'negative_subset'], key),
                                          group=group, metric=metric, mean=float(frame[metric].mean()),
                                          sd_across_outer_folds=float(frame[metric].std(ddof=1))))
            runs.append((group, identity, table))
        for (left, left_id, left_table), (right, right_id, right_table) in combinations(runs, 2):
            def common(identity):
                config = {key: value for key, value in identity['config'].items()
                          if key not in {'feature_profile', 'feature_sets', 'comparisons'}}
                return (identity['shared_protocol_id'], identity['code_fingerprint'], config)
            if common(left_id) != common(right_id):
                skipped.append({'baseline_group': left, 'added_group': right, 'family': family,
                                'reason': 'Protocol, code or common configuration differs.'})
                continue
            keys = ['algorithm', 'condition', 'negative_subset', 'fold']
            for left_feature in left_table.features.unique():
                for right_feature in right_table.features.unique():
                    paired = left_table[left_table.features == left_feature].merge(
                        right_table[right_table.features == right_feature], on=keys,
                        suffixes=('_baseline', '_added'), validate='one_to_one')
                    if len(paired) != len(left_table[left_table.features == left_feature]) or len(paired) != len(right_table[right_table.features == right_feature]):
                        raise ValueError('Incomplete matched comparison: ' + left + ' / ' + right)
                    for row in paired.to_dict('records'):
                        for metric in ('FPR', 'TPR', 'F1', 'ROC_AUC'):
                            records.append({**{key: row[key] for key in keys}, 'baseline_group': left,
                                            'added_group': right, 'baseline_features': left_feature,
                                            'added_features': right_feature, 'metric': metric,
                                            'delta_added_minus_baseline': row[metric + '_added'] - row[metric + '_baseline']})
    pd.DataFrame(summaries, columns=['group', 'algorithm', 'features', 'condition', 'negative_subset',
                                    'metric', 'mean', 'sd_across_outer_folds']).to_csv(output / 'all_group_summary.csv', index=False)
    columns = ['algorithm', 'condition', 'negative_subset', 'fold', 'baseline_group', 'added_group',
               'baseline_features', 'added_features', 'metric', 'delta_added_minus_baseline']
    deltas = pd.DataFrame(records, columns=columns)
    deltas.to_csv(output / 'paired_group_deltas.csv', index=False)
    grouping = [key for key in columns if key not in {'fold', 'delta_added_minus_baseline'}]
    stats = deltas.groupby(grouping).delta_added_minus_baseline.agg(['mean', 'std', 'count']).reset_index()
    stats.to_csv(output / 'paired_group_delta_mean_sd.csv', index=False)
    write_json(output / 'skipped_incompatible_comparisons.json', skipped)
    print('Updated saved comparisons:', output)
