"""Shared product partitions, matching pairs, pixel corruptions and manual metrics."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic checkpoint: an interrupted write cannot be mistaken for completion.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)

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
