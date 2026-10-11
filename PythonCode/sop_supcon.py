"""Supervised contrastive representation learning for unseen-product matching.

Reference: Khosla et al., Supervised Contrastive Learning, NeurIPS 2020,
Equation (2), https://arxiv.org/abs/2004.11362.
Author reference: https://github.com/HobbitLong/SupContrast.

This is an application adaptation: fixed handcrafted/ImageNet descriptors feed
a trainable encoder and disposable projection head. Different clean pictures
of the same SKU provide positives instead of two random pixel augmentations
per image. The learned encoder is retained for cosine matching on unseen SKUs.
It does not reproduce the paper's end-to-end ImageNet classification scores.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

from sop_feature_bank import choose_device
from sop_protocol import manual_metrics, manual_roc_auc, recall_threshold, write_json
from sop_reporting import evaluate_scores, report_results
from sop_calibration import calibrate
from sop_tuning import search_temperatures


class SupConLoss(nn.Module):
    """Mean anchor loss from paper Eq. (2), with self comparisons excluded.

The mean over positives is outside the logarithm. Features may have shape
[samples, dimensions] or [samples, views, dimensions]. No temperature/base-
temperature multiplier is added; this is the paper's objective up to averaging
over anchors rather than summing. Every anchor must have another positive.
"""

    def __init__(self, temperature=0.1):
        super().__init__()
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive and finite.")
        self.temperature = float(temperature)

    def forward(self, features, labels):
        if features.ndim == 3:
            count, views, width = features.shape
            features = features.reshape(count * views, width)
            labels = labels.repeat_interleave(views)
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("SupCon features and labels have incompatible shapes.")
        z = F.normalize(features, dim=1)
        self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
        same_label = labels[:, None].eq(labels[None, :])
        positive_mask = same_label & ~self_mask
        counts = positive_mask.sum(dim=1)
        if torch.any(counts == 0) or labels.unique().numel() < 2:
            raise ValueError("Each anchor needs positives and at least two products in the batch.")
        logits = (z @ z.T) / self.temperature
        denominator = torch.logsumexp(logits.masked_fill(self_mask, -torch.inf), dim=1)
        log_probability = logits - denominator[:, None]
        return -((log_probability * positive_mask).sum(dim=1) / counts).mean()


class MatchingEncoder(nn.Module):
    def __init__(self, input_dim, settings):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, settings["hidden_dim"]), nn.ReLU(),
                                     nn.Linear(settings["hidden_dim"], settings["representation_dim"]))
        self.projector = nn.Sequential(nn.Linear(settings["representation_dim"], settings["hidden_dim"]),
                                       nn.ReLU(), nn.Linear(settings["hidden_dim"], settings["projection_dim"]))

    def encode(self, features):
        return F.normalize(self.encoder(features), dim=1)

    def forward(self, features):
        return F.normalize(self.projector(self.encode(features)), dim=1)


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # Fail rather than silently using a nondeterministic training operation.
    torch.use_deterministic_algorithms(True)


def product_batches(frame, products, settings, rng):
    groups = {int(product): np.flatnonzero(frame.class_id.eq(product).to_numpy()) for product in products}
    if len(groups) < 2 or any(len(rows) < 2 for rows in groups.values()):
        raise ValueError("Contrastive training needs >=2 products with >=2 distinct images each.")
    shuffled = rng.permutation(np.asarray(products, dtype=np.int64))
    size = settings["products_per_batch"]
    batches = [shuffled[start:start + size] for start in range(0, len(shuffled), size)]
    if len(batches[-1]) == 1 and len(batches) > 1:
        last = batches.pop()
        batches[-1] = np.concatenate([batches[-1], last])
    for batch in batches:
        rows, labels = [], []
        for product in batch:
            candidates = groups[int(product)]
            count = min(settings["images_per_product"], len(candidates))
            picked = rng.choice(candidates, size=count, replace=False)
            rows.extend(picked.tolist())
            labels.extend([int(product)] * count)
        yield np.asarray(rows, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def fit_representation(frame, clean_features, products, temperature, settings, seed):
    seed_everything(seed, settings["cpu_threads"])
    device = choose_device(settings["device"], torch)
    rng = np.random.default_rng(seed)
    products = sorted(int(p) for p in products)
    if settings["train_product_cap"] is not None:
        cap = int(settings["train_product_cap"])
        if cap < 2:
            raise ValueError("SupCon train_product_cap must be >=2 or null.")
        products = sorted(int(p) for p in rng.permutation(products)[:cap])
    training_rows = np.flatnonzero(frame.class_id.isin(products).to_numpy())
    training = clean_features[training_rows]
    # Scaling is learned solely from this training product partition.
    mean = training.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = training.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-12] = 1
    model = MatchingEncoder(clean_features.shape[1], settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    criterion = SupConLoss(temperature)
    rng = np.random.default_rng(seed + 1)
    logs = []
    for epoch in range(settings["epochs"]):
        model.train()
        total, anchors = 0.0, 0
        for indexes, labels in product_batches(frame, products, settings, rng):
            batch = torch.from_numpy(((clean_features[indexes] - mean) / scale).astype(np.float32)).to(device)
            labels = torch.from_numpy(labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("SupCon loss is nonfinite.")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("SupCon gradients are nonfinite.")
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indexes)
            anchors += len(indexes)
        logs.append({"epoch": epoch + 1, "mean_loss": total / anchors, "sampled_images": anchors})
        if epoch == 0 or epoch == settings["epochs"] - 1 or (epoch + 1) % 5 == 0:
            print(f"SupCon tau={temperature:g}, epoch {epoch + 1}/{settings['epochs']}, "
                  f"loss={logs[-1]['mean_loss']:.5f}, products={len(products)}", flush=True)
    model.eval()
    return {"model": model, "mean": mean, "scale": scale, "logs": logs,
            "temperature": float(temperature), "training_products": products,
            "input_dimension": clean_features.shape[1], "settings": settings, "device": device}


def encode_rows(fitted, values, indexes):
    outputs = []
    model = fitted["model"]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indexes), 512):
            chunk = indexes[start:start + 512]
            array = ((values[chunk] - fitted["mean"]) / fitted["scale"]).astype(np.float32)
            batch = torch.from_numpy(array).to(fitted["device"])
            outputs.append(model.encode(batch).cpu().numpy())
    return np.concatenate(outputs)


def matching_scores(fitted, pairs, bank, condition="clean"):
    queries = pairs.query_row.to_numpy()
    references = pairs.reference_row.to_numpy()
    query_rows, qmap = np.unique(queries, return_inverse=True)
    reference_rows, rmap = np.unique(references, return_inverse=True)
    query_features = encode_rows(fitted, bank[condition], query_rows)
    reference_features = encode_rows(fitted, bank["clean"], reference_rows)
    scores = np.sum(query_features[qmap] * reference_features[rmap], axis=1)
    if not np.isfinite(scores).all():
        raise RuntimeError("SupCon matching scores are nonfinite.")
    return np.clip(scores, -1, 1)


def save_model(fitted, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in fitted["model"].state_dict().items()}
    checkpoint = {"state_dict": state, "mean": torch.from_numpy(fitted["mean"]),
                  "scale": torch.from_numpy(fitted["scale"]), "input_dimension": fitted["input_dimension"],
                  "settings": fitted["settings"], "temperature": fitted["temperature"],
                  "training_products": fitted["training_products"], "metadata": metadata,
                  "inference": "normalized_learned_encoder_before_discarded_projection_head"}
    temporary = path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    write_json(path.with_name("training_loss.json"), fitted["logs"])


def load_model(path, device="cpu"):
    """Reload a local trusted checkpoint for later matching inference."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    selected_device = choose_device(device, torch)
    model = MatchingEncoder(checkpoint["input_dimension"], checkpoint["settings"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(selected_device).eval()
    return {"model": model, "mean": checkpoint["mean"].numpy(), "scale": checkpoint["scale"].numpy(),
            "device": selected_device, **{key: checkpoint[key] for key in
             ("settings", "temperature", "training_products", "input_dimension")}}


def tune_inner(experiment, context, name, bank, directory):
    config = experiment.config
    settings = config["supcon"]
    def evaluate(temperature):
        fold_auc, labels, scores = [], [], []
        for index, (training_products, validation_products, _, validation_pairs) in enumerate(context["inner"]):
            if set(training_products) & set(validation_products):
                raise ValueError("Inner SupCon products overlap.")
            fitted = fit_representation(experiment.frame, bank["clean"], training_products,
                                        temperature, settings, context["seed"] + index)
            score = matching_scores(fitted, validation_pairs, bank)
            fold_auc.append(manual_roc_auc(validation_pairs.label.to_numpy(), score))
            labels.extend(validation_pairs.label.tolist())
            scores.extend(score.tolist())
            save_model(fitted, Path(directory) / f"tau_{temperature:g}" / f"inner_{index + 1}" / "model.pt",
                       {"features": name, "shared_protocol_id": experiment.protocol_id,
                        "fold": context["fold"], "inner_fold": index + 1})
            del fitted
        return {"mean_auc": float(np.mean(fold_auc)), "fold_auc": fold_auc,
                                    "labels": labels, "scores": scores}
    best, grid, evaluations, history = search_temperatures(evaluate, settings['temperature_grid'],
        settings['maximum_grid_extensions'], settings['temperature_extension_factor'], settings['temperature_bounds'])
    selected = evaluations[best]
    result = {"fold": context["fold"], "algorithm": "SupCon", "features": name,
              "temperature": best, "final_grid": grid, "history": history,
              "boundary_unresolved": best in (grid[0], grid[-1]),
              "candidate_results": {str(t): {k: v for k, v in record.items() if k in {"mean_auc", "fold_auc"}}
                                    for t, record in evaluations.items()},
              "mean_validation_auc": selected['mean_auc'],
              "objective": "Mean clean-validation matching ROC-AUC across inner product folds",
              "threshold_note": "Fixed final encoder calibrated separately on products excluded from tuning and fitting.",
              "adaptation": "fixed_features_learned_encoder_distinct_clean_product_images"}
    write_json(Path(directory) / "tuning.json", result)
    return result


def run_fold(experiment, context, directory):
    rows, records = [], []
    for name, bank in experiment.banks.items():
        print(f"Fold {context['fold']}: SupCon/{name}", flush=True)
        target = Path(directory) / ("SupCon_" + name)
        target.mkdir(parents=True, exist_ok=True)
        tuning = tune_inner(experiment, context, name, bank, target)
        fitted = fit_representation(experiment.frame, bank["clean"], context["train_products"],
                                    tuning["temperature"], experiment.config["supcon"], context["seed"])
        tuning['threshold'] = calibrate(context, matching_scores(fitted, context['calibration_pairs'], bank),
                                       experiment.config, target)
        write_json(target / 'tuning.json', tuning)
        save_model(fitted, target / "model.pt", {"threshold": tuning["threshold"], "features": name,
                   "calibration_products": context['calibration_products'],
                   "shared_protocol_id": experiment.protocol_id, "fold": context["fold"]})
        score_by_condition = {condition: matching_scores(fitted, context["test_pairs"], bank, condition)
                              for condition in bank}
        rows.extend(evaluate_scores(context["test_pairs"], score_by_condition, tuning["threshold"], target,
                                    context["fold"], "SupCon", name))
        records.append(tuning)
        del fitted
    return rows, records


def run_experiment(experiment):
    directory = experiment.result_directory("supcon")
    write_json(directory / "paper_adaptation.json", {
        "paper": "Khosla et al., Supervised Contrastive Learning, NeurIPS 2020, Eq. (2)",
        "paper_url": "https://arxiv.org/abs/2004.11362", "author_code": "https://github.com/HobbitLong/SupContrast",
        "loss": "mean_over_positive_log_probabilities_outside_log; self_excluded; mean_over_anchors",
        "positives": "same_product_class_id", "negatives": "other_products_in_batch",
        "views": "distinct_clean_images_of_same_product; not_two_random_augmentations_of_each_image",
        "encoder": "trainable_MLP_on_fixed_features; external_image_encoder_is_not_finetuned",
        "inference": "learned_encoder_retained; projector_discarded; cosine_and_inner_validation_threshold",
        "evaluation": "unseen_SKU_matching_not_original_ImageNet_classification",
        "mode": experiment.config["mode"], "torch": torch.__version__})
    rows, records = [], []
    folds = list(range(1, len(experiment.folds) + 1))
    if experiment.config["run_official_test"]:
        folds.append("official")
    for fold in folds:
        target = directory / ("official_holdout" if fold == "official" else f"outer_{fold:02d}")
        completed = target / "completed.json"
        if completed.exists():
            saved = json.loads(completed.read_text(encoding="utf-8"))
            current, tuning = saved["rows"], saved["tuning"]
            print(f"SupCon fold {fold}: completed checkpoint reused.")
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
    write_json(directory / "all_tuning_records.json", records)
    report_results(pd.DataFrame(rows), records, directory, experiment.config)
    return directory
