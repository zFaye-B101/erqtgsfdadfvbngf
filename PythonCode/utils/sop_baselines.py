"""Generic LR/RBF-SVM matching using shared features and saved product protocols."""

import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning

from .sop_protocol import manual_metrics, manual_roc_auc, pair_matrix, recall_threshold, write_json
from .sop_reporting import evaluate_scores, report_results
from .sop_calibration import calibrate

def fit_classifier(X, y, algorithm, c, seed):
    # Fit scaling on THIS training fold only; never fit on validation/test.
    mean = X.mean(axis=0, dtype=np.float64)
    scale = X.std(axis=0, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    if algorithm == "LR":
        estimator = LogisticRegression(C=float(c), solver="lbfgs", max_iter=3000,
                                       random_state=seed)
    elif algorithm == "SVM":
        estimator = SVC(C=float(c), kernel="rbf", gamma="scale", probability=False,
                        cache_size=512, random_state=seed)
    else:
        raise ValueError("Unknown classifier: " + algorithm)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator.fit((X - mean) / scale, y)
    convergence = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError("Classifier did not converge. Increase max_iter before reporting results.")
    return {"model": estimator, "mean": mean, "scale": scale,
            "warnings": [str(w.message) for w in captured]}

def scores(model, X):
    return model["model"].decision_function((X - model["mean"]) / model["scale"])


def matrix(pairs, bank, condition="clean"):
    return pair_matrix(pairs, bank, [0, bank["clean"].shape[1]], condition)


def tune_inner(context, bank, algorithm, name, config, directory):
    datasets = [(matrix(train, bank), train.label.to_numpy(), matrix(validation, bank), validation.label.to_numpy())
                for _, _, train, validation in context["inner"]]
    grid = [float(c) for c in config["c_grids"][algorithm]]
    evaluations, history = {}, []
    for stage in range(config["maximum_grid_extensions"] + 1):
        for c in grid:
            if c in evaluations:
                continue
            fold_auc, all_y, all_score = [], [], []
            for fold, (train_x, train_y, val_x, val_y) in enumerate(datasets):
                model = fit_classifier(train_x, train_y, algorithm, c, context["seed"] + fold)
                val_score = scores(model, val_x)
                fold_auc.append(manual_roc_auc(val_y, val_score))
                all_y.extend(val_y.tolist())
                all_score.extend(val_score.tolist())
            evaluations[c] = {"fold_auc": fold_auc, "mean_auc": float(np.mean(fold_auc)),
                              "y": all_y, "score": all_score}
        best = max([grid[1], grid[0], grid[2]], key=lambda c: evaluations[c]["mean_auc"])
        history.append({"stage": stage, "grid": grid.copy(), "selected_C": best,
                        "mean_validation_auc": {str(c): evaluations[c]["mean_auc"] for c in grid}})
        if best == grid[1] or stage == config["maximum_grid_extensions"]:
            break
        grid = [grid[0] / 10, grid[0], grid[1]] if best == grid[0] else [grid[1], grid[2], grid[2] * 10]
    chosen = evaluations[best]
    result = {"fold": context["fold"], "algorithm": algorithm, "features": name,
              "C": best, "final_grid": grid,
              "boundary_unresolved": best != grid[1], "history": history,
              "mean_validation_auc": chosen['mean_auc'],
              "objective": "Mean clean-validation ROC-AUC across product-grouped inner folds",
              "threshold_note": "Threshold is calibrated separately after final fitting; calibration products are excluded from tuning."}
    write_json(Path(directory) / "tuning.json", result)
    return result


def run_fold(experiment, context, directory):
    rows, records = [], []
    for algorithm in ("LR", "SVM"):
        for name, bank in experiment.banks.items():
            print(f"Fold {context['fold']}: {algorithm}/{name}", flush=True)
            target = Path(directory) / (algorithm + "_" + name)
            target.mkdir(parents=True, exist_ok=True)
            tuning = tune_inner(context, bank, algorithm, name, experiment.config, target)
            model = fit_classifier(matrix(context["train_pairs"], bank), context["train_pairs"].label.to_numpy(),
                                   algorithm, tuning["C"], context["seed"])
            tuning['threshold'] = calibrate(context, scores(model, matrix(context['calibration_pairs'], bank)),
                                            experiment.config, target)
            write_json(target / 'tuning.json', tuning)
            with (target / "model.pkl").open("wb") as stream:
                pickle.dump({**model, "threshold": tuning["threshold"], "C": tuning["C"],
                             "fitting_products": context['train_products'], "calibration_products": context['calibration_products'],
                             "features": name, "input_dimension": bank["clean"].shape[1]}, stream)
            score_by_condition = {condition: scores(model, matrix(context["test_pairs"], bank, condition))
                                  for condition in bank}
            rows.extend(evaluate_scores(context["test_pairs"], score_by_condition, tuning["threshold"],
                                        target, context["fold"], algorithm, name))
            records.append(tuning)
    return rows, records


def run_experiment(experiment):
    directory = experiment.result_directory("baselines")
    rows, records = [], []
    folds = list(range(1, len(experiment.folds) + 1))
    if experiment.config["run_official_test"]:
        folds.append("official")
    for fold in folds:
        target = directory / ("official_holdout" if fold == "official" else f"outer_{fold:02d}")
        completed = target / "completed.json"
        if completed.exists():
            import json
            saved = json.loads(completed.read_text(encoding="utf-8"))
            current, tuning = saved["rows"], saved["tuning"]
            print(f"Baseline fold {fold}: completed checkpoint reused.")
        else:
            start = time.time()
            current, tuning = run_fold(experiment, experiment.context(fold), target)
            write_json(completed, {"rows": current, "tuning": tuning, "seconds": time.time() - start})
        if fold == "official":
            pd.DataFrame(current).to_csv(target / "metrics.csv", index=False)
        else:
            rows.extend(current)
            records.extend(tuning)
            pd.DataFrame(rows).to_csv(directory / "outer_fold_metrics.csv", index=False)
    results = pd.DataFrame(rows)
    write_json(directory / "all_tuning_records.json", records)
    report_results(results, records, directory, experiment.config)
    return directory
