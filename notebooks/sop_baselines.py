"""Traditional SOP pair matching: manual product-grouped nested CV and metrics.

Only the classifier implementations come from sklearn. Splits, tuning,
standardisation, thresholds, metrics and error bars are implemented here.
"""

from pathlib import Path
import hashlib
import json
import pickle
import platform
import time
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
import sklearn

from sop_features import image_feature_vector


DEFAULT_CONFIG = {
    "seed": 20261011,
    "minimum_train_images": 10000,
    "use_all_official_train": False,
    "outer_folds": 10,
    "inner_folds": 3,
    "train_anchor_cap": 1000,
    "evaluation_anchor_cap": 500,
    "negative_candidates": 50,
    "target_validation_recall": 0.95,
    "c_grids": {"LR": [0.01, 0.1, 1.0], "SVM": [0.1, 1.0, 10.0]},
    "maximum_grid_extensions": 2,
    "conditions": {
        "clean": {},
        "occlusion25": {"area_fraction": 0.25, "fill": 127},
        "noise005": {"sigma": 0.05},
        "blur15": {"radius": 1.5},
        "rotation15": {"degrees": 15.0},
    },
    "feature_sets": {"HSV48": [0, 48], "HSV_HOG192": [0, 192]},
    "run_official_test": True,
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic checkpoint: an interrupted write cannot be mistaken for completion.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_preprocessed(data_dir):
    data_dir = Path(data_dir).resolve()
    manifest = pd.read_csv(data_dir / "clean_manifest.csv")
    with np.load(data_dir / "image_features_hsv_hog.npz", allow_pickle=False) as z:
        X = z["X"].astype(np.float32)
        ids = z["image_ids"].astype(np.int64)
        for column, key in [("image_id", "image_ids"), ("class_id", "class_ids"),
                            ("super_class_id", "super_class_ids"),
                            ("source_split", "source_split")]:
            if not np.array_equal(manifest[column].to_numpy(), z[key]):
                raise ValueError("Manifest and feature cache disagree: " + column)
    if X.shape != (len(manifest), 192) or not np.isfinite(X).all():
        raise ValueError("Expected a finite N x 192 feature matrix from preprocessing script 01.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate image IDs.")
    if not set(manifest.source_split.unique()) <= {"train", "test"}:
        raise ValueError("Unknown official split.")
    if (manifest.groupby("class_id").source_split.nunique() != 1).any():
        raise ValueError("A product occurs in both official splits.")
    if (manifest.groupby("class_id").super_class_id.nunique() != 1).any():
        raise ValueError("Inconsistent product/category labels.")
    if manifest.groupby("class_id").size().min() < 2:
        raise ValueError("Run singleton removal in preprocessing script 01 first.")
    if (manifest[["original_width", "original_height"]] < 100).any().any():
        raise ValueError("Original images below the assignment's minimum size.")
    return manifest, X


def select_train_products(manifest, config):
    train = manifest[manifest.source_split.eq("train")]
    if len(train) < config["minimum_train_images"]:
        raise ValueError("Not enough eligible original training images.")
    if config["use_all_official_train"]:
        return sorted(int(x) for x in train.class_id.unique())
    rng = np.random.default_rng(config["seed"])
    products = train.groupby("class_id").agg(
        category=("super_class_id", "first"), images=("image_id", "size"))
    queues = {int(cat): list(rng.permutation(group.index.to_numpy()).astype(int))
              for cat, group in products.groupby("category")}
    selected, image_count = [], 0
    # Round-robin sampling across categories; retain ALL images of each chosen SKU.
    while image_count < config["minimum_train_images"]:
        moved = False
        for cat in sorted(queues):
            if queues[cat]:
                product = queues[cat].pop()
                selected.append(int(product))
                image_count += int(products.loc[product, "images"])
                moved = True
        if not moved:
            raise ValueError("Cannot reach the required image count.")
    return sorted(selected)


def product_folds(manifest, products, k, seed):
    """Manual category-stratified k-fold partition of product IDs, not images."""
    table = manifest[manifest.class_id.isin(products)].groupby("class_id").agg(
        category=("super_class_id", "first"))
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cat, group in table.groupby("category", sort=True):
        if len(group) < k:
            raise ValueError(f"Category {cat} has fewer than {k} selected products.")
        shuffled = rng.permutation(group.index.to_numpy())
        offset = int(rng.integers(k))
        for position, product in enumerate(shuffled):
            folds[(position + offset) % k].append(int(product))
    folds = [sorted(f) for f in folds]
    flat = [p for fold in folds for p in fold]
    if sorted(flat) != sorted(products) or len(flat) != len(set(flat)):
        raise ValueError("Invalid product fold partition.")
    return folds


def content_box(row, size):
    """Recover the content rectangle inside preprocessing script 01's padding."""
    scale = min(size / row.original_width, size / row.original_height)
    width = max(1, int(round(row.original_width * scale)))
    height = max(1, int(round(row.original_height * scale)))
    left, top = (size - width) // 2, (size - height) // 2
    return left, top, left + width, top + height


def corrupt_image(image, row, condition, parameters, seed):
    """One corruption at a time, on the query; reference stays clean."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(row.image_id)]))
    if condition == "clean":
        return image.copy()
    image = image.convert("RGB")
    # Apply corruption only within the original content, keeping the padding fixed.
    box = content_box(row, image.width)
    content = image.crop(box)
    if condition.startswith("occlusion"):
        arr = np.array(content)
        height, width = arr.shape[:2]
        side_fraction = np.sqrt(parameters["area_fraction"])
        mw = max(1, int(round(width * side_fraction)))
        mh = max(1, int(round(height * side_fraction)))
        x = int(rng.integers(0, width - mw + 1))
        y = int(rng.integers(0, height - mh + 1))
        arr[y:y + mh, x:x + mw] = parameters["fill"]
        content = Image.fromarray(arr)
    elif condition.startswith("noise"):
        arr = np.asarray(content, dtype=np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0, parameters["sigma"], arr.shape), 0, 1)
        content = Image.fromarray(np.rint(arr * 255).astype(np.uint8))
    elif condition.startswith("blur"):
        content = content.filter(ImageFilter.GaussianBlur(parameters["radius"]))
    elif condition.startswith("rotation"):
        # Fixed +15 degrees, same canvas; cropping is part of this test condition.
        content = content.rotate(parameters["degrees"],
                                 resample=Image.Resampling.BILINEAR,
                                 expand=False, fillcolor=(255, 255, 255))
    else:
        raise ValueError("Unknown corruption: " + condition)
    result = image.copy()
    result.paste(content, (box[0], box[1]))
    return result


def prepare_features(manifest, X, data_dir, output_dir, selected, config):
    """Cache non-learned features for selected train + optional official test."""
    mask = manifest.class_id.isin(selected)
    if config["run_official_test"]:
        mask |= manifest.source_split.eq("test")
    original_rows = np.flatnonzero(mask.to_numpy())
    frame = manifest.iloc[original_rows].copy().reset_index(drop=True)
    feature_bank = {"clean": X[original_rows].copy()}
    # Only queries need corrupted features. Keep the full official-test reference
    # pool, but avoid corrupting 60k images when only 500 anchor SKUs are tested.
    query_products = set(selected)
    if config["run_official_test"]:
        test_products = np.asarray(sorted(int(p) for p in frame.loc[
            frame.source_split.eq("test"), "class_id"].unique()), dtype=np.int64)
        official_rng = np.random.default_rng(config["seed"] + 50002)
        query_products.update(int(p) for p in official_rng.permutation(test_products)[
            :config["evaluation_anchor_cap"]])
    query_mask = frame.class_id.isin(query_products).to_numpy()
    cache_dir = Path(output_dir) / "feature_cache"
    cache_dir.mkdir(exist_ok=True)
    np.save(cache_dir / "image_ids.npy", frame.image_id.to_numpy())
    for number, (condition, parameters) in enumerate(config["conditions"].items()):
        if condition == "clean":
            continue
        cache_path = cache_dir / (condition + ".npy")
        if cache_path.exists():
            features = np.load(cache_path)
            if features.shape != (len(frame), 192) or not np.isfinite(features).all():
                raise ValueError("Invalid feature cache: " + str(cache_path))
        else:
            features = feature_bank["clean"].copy()
            completed_queries = 0
            for i, row in enumerate(frame.itertuples(index=False)):
                if not query_mask[i]:
                    continue
                path = (Path(data_dir) / row.processed_relative_path).resolve()
                if not path.is_relative_to(Path(data_dir).resolve()):
                    raise ValueError("Image path escapes the data directory.")
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                if image.size != (224, 224):
                    raise ValueError("Expected preprocessing script 01's 224 x 224 images.")
                altered = corrupt_image(image, row, condition, parameters,
                                        config["seed"] + number * 100003)
                features[i] = image_feature_vector(altered)
                completed_queries += 1
                if completed_queries % 2000 == 0:
                    print(f"{condition}: {completed_queries}/{int(query_mask.sum())} query images", flush=True)
            temporary = cache_dir / (condition + ".tmp.npy")
            np.save(temporary, features)
            temporary.replace(cache_path)
        feature_bank[condition] = features
    frame["corrupted_query_features_available"] = query_mask
    frame.to_csv(Path(output_dir) / "experiment_manifest.csv", index=False)
    return frame, feature_bank


def make_pairs(frame, clean, products, cap, candidates, seed):
    """Positive + same-category random negative + colour-selected hard negative.

    Hard negatives are selected using CLEAN HSV48 for all methods, independent
    of trained models. This is a colour-similarity proxy, not semantic certainty.
    """
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(frame.class_id.isin(products).to_numpy())
    sku = frame.class_id.to_numpy()
    category = frame.super_class_id.to_numpy()
    groups = {int(p): rows[sku[rows] == p] for p in sorted(products)}
    category_rows = {int(c): rows[category[rows] == c] for c in np.unique(category[rows])}
    anchors = rng.permutation(np.asarray(sorted(products), dtype=np.int64))[:cap]
    pairs = []
    used_negatives = set()
    for product in anchors:
        query, reference = rng.choice(groups[int(product)], size=2, replace=False)
        pool = category_rows[int(category[query])]
        pool = pool[sku[pool] != product]
        if len(pool) < 2:
            raise ValueError("Need at least two other-category-pool images per anchor.")
        available = np.asarray([r for r in pool
                                if tuple(sorted((int(query), int(r)))) not in used_negatives])
        if len(available) < 2:
            continue
        random_reference = int(rng.choice(available))
        remaining = available[available != random_reference]
        sampled = rng.choice(remaining, size=min(candidates, len(remaining)), replace=False)
        distances = np.sum((clean[sampled, :48] - clean[query, :48]) ** 2, axis=1)
        hard_reference = int(sampled[np.argmin(distances)])
        pairs.extend([(int(query), int(reference), 1, "positive"),
                      (int(query), random_reference, 0, "random"),
                      (int(query), hard_reference, 0, "colour_hard")])
        used_negatives.add(tuple(sorted((int(query), random_reference))))
        used_negatives.add(tuple(sorted((int(query), hard_reference))))
    result = pd.DataFrame(pairs, columns=["query_row", "reference_row", "label", "kind"])
    if result.empty or result.label.nunique() != 2:
        raise ValueError("Cannot construct valid matching pairs.")
    q, r = result.query_row.to_numpy(), result.reference_row.to_numpy()
    if not np.array_equal((sku[q] == sku[r]).astype(int), result.label.to_numpy()):
        raise ValueError("Pair label error.")
    result["query_image_id"] = frame.image_id.to_numpy()[q]
    result["reference_image_id"] = frame.image_id.to_numpy()[r]
    result["query_product_id"] = sku[q]
    result["reference_product_id"] = sku[r]
    result["category"] = category[q]
    result["selection_hsv_l2"] = np.sqrt(np.sum((clean[q, :48] - clean[r, :48]) ** 2, axis=1))
    return result


def pair_matrix(pairs, bank, feature_range, condition="clean"):
    start, end = feature_range
    q, r = pairs.query_row.to_numpy(), pairs.reference_row.to_numpy()
    return np.abs(bank[condition][q, start:end] - bank["clean"][r, start:end])


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


def manual_roc_auc(y, score):
    """Mann-Whitney rank statistic with average ranks for tied scores."""
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        raise ValueError("ROC-AUC needs both labels.")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(y), dtype=float)
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) /
                 (positive * negative))


def manual_average_precision(y, score):
    order = np.argsort(-np.asarray(score), kind="stable")
    sorted_y, sorted_score = np.asarray(y)[order], np.asarray(score)[order]
    ends = np.r_[np.flatnonzero(sorted_score[:-1] != sorted_score[1:]), len(y) - 1]
    tp = np.cumsum(sorted_y)[ends].astype(float)
    precision = tp / (ends + 1)
    recall = tp / sorted_y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def recall_threshold(y, score, target):
    positive_scores = np.sort(np.asarray(score)[np.asarray(y) == 1])[::-1]
    if len(positive_scores) == 0:
        raise ValueError("No positive validation pairs.")
    rank = max(1, int(np.ceil(target * len(positive_scores))))
    return float(positive_scores[rank - 1])


def manual_metrics(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    recall, fpr = tp / (tp + fn), fp / (fp + tn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": (tp + tn) / len(y), "precision": precision,
            "TPR": recall, "FPR": fpr, "F1": f1,
            "balanced_accuracy": (recall + 1 - fpr) / 2,
            "ROC_AUC": manual_roc_auc(y, score),
            "average_precision": manual_average_precision(y, score)}


def tune_inner(frame, bank, products, algorithm, feature_name, config, directory, seed):
    """Explicit 3-fold tuning; optional range extension uses inner data ONLY."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    folds = product_folds(frame, products, config["inner_folds"], seed)
    write_json(directory / "inner_product_folds.json", folds)
    datasets = []
    for fold, validation_products in enumerate(folds):
        train_products = sorted(set(products) - set(validation_products))
        assert not set(train_products) & set(validation_products)
        tr = make_pairs(frame, bank["clean"], train_products, config["train_anchor_cap"],
                        config["negative_candidates"], seed + fold * 11 + 1)
        va = make_pairs(frame, bank["clean"], validation_products,
                        config["evaluation_anchor_cap"], config["negative_candidates"],
                        seed + fold * 11 + 2)
        tr.to_csv(directory / f"inner{fold + 1}_train_pairs.csv", index=False)
        va.to_csv(directory / f"inner{fold + 1}_validation_pairs.csv", index=False)
        datasets.append((pair_matrix(tr, bank, config["feature_sets"][feature_name]),
                         tr.label.to_numpy(),
                         pair_matrix(va, bank, config["feature_sets"][feature_name]),
                         va.label.to_numpy()))
    grid = [float(c) for c in config["c_grids"][algorithm]]
    if len(grid) != 3 or not (0 < grid[0] < grid[1] < grid[2]):
        raise ValueError("Use exactly three increasing positive C values.")
    evaluations, history = {}, []
    for stage in range(config["maximum_grid_extensions"] + 1):
        for c in grid:
            if c not in evaluations:
                fold_auc, all_y, all_score = [], [], []
                for fold, (train_X, train_y, val_X, val_y) in enumerate(datasets):
                    model = fit_classifier(train_X, train_y, algorithm, c, seed + fold)
                    val_score = scores(model, val_X)
                    fold_auc.append(manual_roc_auc(val_y, val_score))
                    all_y.extend(val_y.tolist())
                    all_score.extend(val_score.tolist())
                evaluations[c] = {"fold_auc": fold_auc, "mean_auc": float(np.mean(fold_auc)),
                                  "y": all_y, "score": all_score}
        # Deterministic tie handling prefers the middle C, without altering scores.
        best = max([grid[1], grid[0], grid[2]], key=lambda c: evaluations[c]["mean_auc"])
        history.append({"stage": stage, "grid": grid.copy(), "selected_C": best,
                        "mean_validation_auc": {str(c): evaluations[c]["mean_auc"] for c in grid}})
        if best == grid[1] or stage == config["maximum_grid_extensions"]:
            break
        grid = [grid[0] / 10, grid[0], grid[1]] if best == grid[0] else [grid[1], grid[2], grid[2] * 10]
    chosen = evaluations[best]
    threshold = recall_threshold(chosen["y"], chosen["score"], config["target_validation_recall"])
    result = {"algorithm": algorithm, "features": feature_name, "C": best,
              "threshold": threshold, "final_grid": grid,
              "boundary_unresolved": best != grid[1], "history": history,
              "validation_metrics": manual_metrics(chosen["y"], chosen["score"], threshold),
              "objective": "Mean clean-validation ROC-AUC over 3 product folds",
              "threshold_note": "Pooled inner out-of-fold clean scores; test TPR need not equal 0.95."}
    write_json(directory / "tuning.json", result)
    return result


def evaluate(model, threshold, pairs, bank, feature_range, output_dir, fold, algorithm, feature_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    y = pairs.label.to_numpy()
    kinds = pairs.kind.to_numpy()
    for condition in bank:
        score = scores(model, pair_matrix(pairs, bank, feature_range, condition))
        prediction = pairs.copy()
        prediction["condition"] = condition
        prediction["score"] = score
        prediction["threshold"] = threshold
        prediction["prediction"] = (score >= threshold).astype(int)
        prediction.to_csv(output_dir / f"predictions_{condition}.csv.gz", index=False)
        for subset in ["all", "random", "colour_hard"]:
            mask = np.ones(len(y), dtype=bool) if subset == "all" else ((kinds == "positive") | (kinds == subset))
            row = {"fold": fold, "algorithm": algorithm, "features": feature_name,
                   "condition": condition, "negative_subset": subset,
                   "threshold": threshold, "pair_count": int(mask.sum())}
            row.update(manual_metrics(y[mask], score[mask], threshold))
            results.append(row)
    return results


def run_experiment(data_dir, output_dir, config=None):
    config = json.loads(json.dumps(DEFAULT_CONFIG if config is None else config))
    if config["outer_folds"] < 10 or config["inner_folds"] < 3:
        raise ValueError("Formal protocol requires >=10 outer and >=3 inner folds.")
    if config["minimum_train_images"] < 10000:
        raise ValueError("Keep at least 10000 eligible original images.")
    if list(config["conditions"])[0] != "clean":
        raise ValueError("clean must be the first condition.")
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir == data_dir or output_dir.is_relative_to(data_dir):
        raise ValueError("Keep experiment outputs outside processed_sop.")
    manifest, X = load_preprocessed(data_dir)
    selected = select_train_products(manifest, config)
    folds = product_folds(manifest, selected, config["outer_folds"], config["seed"] + 100)
    digest = hashlib.sha256(X.tobytes() + manifest.to_csv(index=False).encode("utf-8")).hexdigest()
    source_digest = hashlib.sha256(Path(__file__).read_bytes() +
                                   Path(__file__).with_name("sop_features.py").read_bytes()).hexdigest()
    protocol = {"config": config, "data_fingerprint": digest, "code_fingerprint": source_digest,
                "selected_train_products": selected, "outer_product_folds": folds,
                "selected_train_images": int(manifest.class_id.isin(selected).sum()),
                "data_dir": str(data_dir), "python": platform.python_version(),
                "numpy": np.__version__, "sklearn": sklearn.__version__}
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Protocol/data/code changed. Choose a NEW output directory.")
        print("Identical protocol: reusing feature caches and completed outer folds.")
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError("Output directory is nonempty but has no protocol.json. Choose a NEW directory.")
        write_json(protocol_path, protocol)
    print(f"Selected {protocol['selected_train_images']} images / {len(selected)} training products.")
    frame, bank = prepare_features(manifest, X, data_dir, output_dir, selected, config)
    # Rows used for pairs are fixed and saved; all algorithms see the same pairs.
    rows, tuning_records = [], []
    for fold, test_products in enumerate(folds, start=1):
        directory = output_dir / f"outer_{fold:02d}"
        completed = directory / "completed.json"
        if completed.exists():
            previous = json.loads(completed.read_text(encoding="utf-8"))
            rows.extend(previous["rows"])
            tuning_records.extend(previous["tuning"])
            print(f"Outer fold {fold}/{config['outer_folds']}: checkpoint loaded.", flush=True)
            continue
        started = time.time()
        train_products = sorted(set(selected) - set(test_products))
        assert not set(train_products) & set(test_products)
        directory.mkdir(exist_ok=True)
        train_pairs = make_pairs(frame, bank["clean"], train_products,
                                 config["train_anchor_cap"], config["negative_candidates"],
                                 config["seed"] + fold * 1000 + 1)
        test_pairs = make_pairs(frame, bank["clean"], test_products,
                                config["evaluation_anchor_cap"], config["negative_candidates"],
                                config["seed"] + fold * 1000 + 2)
        train_pairs.to_csv(directory / "train_pairs.csv", index=False)
        test_pairs.to_csv(directory / "test_pairs.csv", index=False)
        current_rows, current_tuning = [], []
        for algorithm in ["LR", "SVM"]:
            for feature_name, feature_range in config["feature_sets"].items():
                print(f"Fold {fold}/{config['outer_folds']}: {algorithm} {feature_name} tuning...", flush=True)
                name = algorithm + "_" + feature_name
                # Same inner folds and pair seeds for EVERY algorithm / feature set.
                tuning = tune_inner(frame, bank, train_products, algorithm, feature_name,
                                    config, directory / name,
                                    config["seed"] + fold * 1000 + 100)
                tuning["fold"] = fold
                current_tuning.append(tuning)
                model = fit_classifier(pair_matrix(train_pairs, bank, feature_range),
                                       train_pairs.label.to_numpy(), algorithm,
                                       tuning["C"], config["seed"] + fold)
                with (directory / name / "model.pkl").open("wb") as stream:
                    pickle.dump({**model, "threshold": tuning["threshold"],
                                 "feature_range": feature_range, "C": tuning["C"]}, stream)
                current_rows.extend(evaluate(model, tuning["threshold"], test_pairs, bank,
                                             feature_range, directory / name, fold,
                                             algorithm, feature_name))
        write_json(completed, {"rows": current_rows, "tuning": current_tuning,
                               "seconds": time.time() - started})
        rows.extend(current_rows)
        tuning_records.extend(current_tuning)
        pd.DataFrame(rows).to_csv(output_dir / "outer_fold_metrics.csv", index=False)
        print(f"Outer fold {fold} completed in {(time.time() - started) / 60:.1f} minutes.", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "outer_fold_metrics.csv", index=False)
    write_json(output_dir / "all_tuning_records.json", tuning_records)
    report_results(results, tuning_records, output_dir)
    if config["run_official_test"]:
        run_final_official_test(frame, bank, selected, config, output_dir)
    print("Completed. Results:", output_dir)
    return results


def report_results(results, tuning_records, output_dir):
    import matplotlib.pyplot as plt
    metrics = ["accuracy", "precision", "TPR", "FPR", "F1", "balanced_accuracy",
               "ROC_AUC", "average_precision"]
    keys = ["algorithm", "features", "condition", "negative_subset"]
    summary = []
    for key, group in results.groupby(keys, sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            mean = float(values.sum() / len(values))
            sd = float(np.sqrt(np.sum((values - mean) ** 2) / (len(values) - 1)))
            summary.append({**dict(zip(keys, key)), "metric": metric,
                            "mean": mean, "sd_across_outer_folds": sd,
                            "outer_fold_count": len(values)})
    pd.DataFrame(summary).to_csv(Path(output_dir) / "summary_mean_sd.csv", index=False)
    # Paired ablation contrasts: SAME fold, pairs, corruption and classifier.
    deltas = []
    for (algorithm, condition, subset), group in results.groupby(
            ["algorithm", "condition", "negative_subset"]):
        for metric in ["FPR", "TPR", "F1", "ROC_AUC"]:
            pivot = group.pivot(index="fold", columns="features", values=metric)
            if {"HSV48", "HSV_HOG192"} <= set(pivot.columns):
                for fold, value in (pivot.HSV_HOG192 - pivot.HSV48).items():
                    deltas.append({"algorithm": algorithm, "condition": condition,
                                   "negative_subset": subset, "metric": metric,
                                   "fold": int(fold), "delta_local_minus_global": float(value)})
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(Path(output_dir) / "paired_feature_deltas.csv", index=False)
    delta_summary = []
    for key, group in delta_frame.groupby(["algorithm", "condition", "negative_subset", "metric"]):
        values = group.delta_local_minus_global.to_numpy()
        mean = float(values.sum() / len(values))
        sd = float(np.sqrt(np.sum((values - mean) ** 2) / (len(values) - 1)))
        delta_summary.append({**dict(zip(["algorithm", "condition", "negative_subset", "metric"], key)),
                              "mean_delta": mean, "sd_across_outer_folds": sd})
    pd.DataFrame(delta_summary).to_csv(Path(output_dir) / "paired_delta_mean_sd.csv", index=False)
    table = pd.DataFrame(summary)
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(exist_ok=True)
    for subset in ["random", "colour_hard"]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        conditions = list(results.condition.unique())
        for axis, metric in zip(axes.ravel(), ["FPR", "TPR", "F1", "ROC_AUC"]):
            for algorithm in ["LR", "SVM"]:
                for feature_name in ["HSV48", "HSV_HOG192"]:
                    part = table[(table.algorithm == algorithm) & (table.features == feature_name)
                                 & (table.negative_subset == subset) & (table.metric == metric)]
                    part = part.set_index("condition").loc[conditions]
                    axis.errorbar(np.arange(len(conditions)), part["mean"],
                                  yerr=part.sd_across_outer_folds, marker="o", capsize=3,
                                  label=algorithm + " / " + feature_name)
            axis.set_xticks(np.arange(len(conditions)), conditions, rotation=20)
            axis.set_title(metric + " (mean +/- 1 outer-fold SD)")
            axis.set_ylim(-0.05, 1.05)
            axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Traditional features / " + subset + " negatives")
        fig.savefig(figure_dir / (subset + "_robustness.png"), dpi=180)
        plt.close(fig)
    grid_audit = []
    for (algorithm, features), group in pd.DataFrame(tuning_records).groupby(["algorithm", "features"]):
        middle_count = sum(r["C"] == r["final_grid"][1] for r in group.to_dict("records"))
        boundary_count = len(group) - middle_count
        grid_audit.append({"algorithm": algorithm, "features": features,
                           "middle_selected_folds": middle_count, "boundary_selected_folds": boundary_count,
                           "middle_is_majority": middle_count > len(group) / 2})
    write_json(Path(output_dir) / "grid_requirement_audit.json", grid_audit)
    for row in grid_audit:
        if not row["middle_is_majority"]:
            print("ATTENTION: C range remains unresolved:", row,
                  "Use inner validation evidence to extend ranges before a formal report.")


def run_final_official_test(frame, bank, products, config, output_dir):
    """Supplementary single official holdout; NEVER the source of error bars."""
    directory = Path(output_dir) / "official_holdout"
    if (directory / "completed.json").exists():
        print("Official holdout checkpoint loaded; not reselecting models.")
        return
    directory.mkdir(exist_ok=True)
    test_products = sorted(int(p) for p in frame.loc[frame.source_split.eq("test"), "class_id"].unique())
    if not test_products or set(products) & set(test_products):
        raise ValueError("Invalid official test partition.")
    train_pairs = make_pairs(frame, bank["clean"], products, config["train_anchor_cap"],
                             config["negative_candidates"], config["seed"] + 50001)
    test_pairs = make_pairs(frame, bank["clean"], test_products, config["evaluation_anchor_cap"],
                            config["negative_candidates"], config["seed"] + 50002)
    train_pairs.to_csv(directory / "train_pairs.csv", index=False)
    test_pairs.to_csv(directory / "test_pairs.csv", index=False)
    rows = []
    for algorithm in ["LR", "SVM"]:
        for feature_name, feature_range in config["feature_sets"].items():
            name = algorithm + "_" + feature_name
            print("Final official holdout:", name, flush=True)
            tuning = tune_inner(frame, bank, products, algorithm, feature_name, config,
                                directory / name, config["seed"] + 51000)
            model = fit_classifier(pair_matrix(train_pairs, bank, feature_range),
                                   train_pairs.label.to_numpy(), algorithm, tuning["C"], config["seed"])
            with (directory / name / "model.pkl").open("wb") as stream:
                pickle.dump({**model, "threshold": tuning["threshold"],
                             "feature_range": feature_range, "C": tuning["C"]}, stream)
            rows.extend(evaluate(model, tuning["threshold"], test_pairs, bank, feature_range,
                                 directory / name, "official", algorithm, feature_name))
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    write_json(directory / "completed.json", {"rows": rows, "note":
               "One supplementary holdout. Do not tune from these results or derive CV error bars here."})
