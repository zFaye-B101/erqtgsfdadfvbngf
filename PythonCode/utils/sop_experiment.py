"""Feature-independent protocol shared by LR, SVM and SupCon.

All pair selection uses a fixed clean HSV colour descriptor, irrespective of
the feature being evaluated. Changing a feature does not change test pairs.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sop_config import validate_config
from .sop_calibration import split_calibration_products
from .sop_feature_bank import build_banks, dataset_digest, extract_cached, file_digest
from .sop_protocol import make_pairs, product_folds, select_train_products, write_json


def load_manifest(data_dir):
    root = Path(data_dir)
    summary_path = root / "audit_summary.json"
    if not summary_path.exists():
        raise ValueError("Preprocessing is incomplete: audit_summary.json is missing.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("complete", True) is not True:
        raise ValueError("Preprocessing has not completed.")
    frame = pd.read_csv(root / "clean_manifest.csv")
    required = {"image_id", "class_id", "super_class_id", "source_split",
                "original_width", "original_height", "processed_relative_path"}
    if not required <= set(frame):
        raise ValueError("Missing preprocessing manifest columns.")
    if frame.empty or frame.image_id.duplicated().any():
        raise ValueError("Empty manifest or duplicate image IDs.")
    if not set(frame.source_split) <= {"train", "test"}:
        raise ValueError("Unknown official split.")
    grouped = frame.groupby("class_id")
    if (grouped.source_split.nunique() != 1).any() or (grouped.super_class_id.nunique() != 1).any():
        raise ValueError("Inconsistent product split/category labels.")
    if grouped.size().min() < 2 or (frame[["original_width", "original_height"]] < 100).any().any():
        raise ValueError("Run script 01's size and singleton filtering first.")
    if len(frame) != summary["clean_images"]:
        raise ValueError("Preprocessing summary and manifest disagree.")
    return frame


def guard_directory(directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "protocol.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != identity:
            raise ValueError("Configuration, data or code changed. Choose a new run/protocol directory: " + str(directory))
    else:
        if any(directory.iterdir()):
            raise ValueError("Nonempty directory has no protocol.json: " + str(directory))
        write_json(target, identity)


class Experiment:
    def __init__(self, config):
        validate_config(config)
        self.config = config
        manifest = load_manifest(config["data_dir"])
        self.dataset_id = dataset_digest(config["data_dir"], manifest)
        self.selected = select_train_products(manifest, config)
        mask = manifest.class_id.isin(self.selected)
        if config["run_official_test"]:
            mask |= manifest.source_split.eq("test")
        self.frame = manifest.loc[mask].reset_index(drop=True)
        self.folds = product_folds(self.frame, self.selected, config["outer_folds"], config["seed"] + 100)
        fields = ("mode", "seed", "minimum_train_images", "use_all_official_train",
                  "outer_folds", "inner_folds", "train_anchor_cap", "evaluation_anchor_cap",
                  "negative_candidates", "target_validation_recall", "conditions", "run_official_test", "calibration_fraction")
        self.protocol = {"protocol_version": 3, "dataset_id": self.dataset_id,
                         "settings": {key: config[key] for key in fields},
                         "selected_train_products": self.selected, "outer_product_folds": self.folds,
                         "frame_image_ids": self.frame.image_id.tolist(),
                         "selected_train_images": int(manifest.class_id.isin(self.selected).sum()),
                         "hard_negative_rule": "clean_content_HSV48_same_category_configured_candidates",
                         "protocol_source": file_digest([Path(__file__).with_name("sop_protocol.py"), Path(__file__).with_name("sop_calibration.py"), __file__])}
        self.protocol_id = hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()
        self.directory = Path(config["protocol_dir"])
        guard_directory(self.directory, self.protocol)
        self.query_mask = self.frame.class_id.isin(self.selected).to_numpy()
        if config["run_official_test"]:
            products = sorted(int(p) for p in self.frame.loc[self.frame.source_split.eq("test"), "class_id"].unique())
            rng = np.random.default_rng(config["seed"] + 50002)
            queries = rng.permutation(products)[:config["evaluation_anchor_cap"]]
            self.query_mask |= self.frame.class_id.isin(queries).to_numpy()
        # Colour matching remains independent of the experimental representation.
        self.hard_features = extract_cached(self.frame, config["data_dir"], config,
                                           self.dataset_id, "hsv_hog_content", "clean", self.query_mask)[:, :48]
        self.banks = build_banks(self.frame, config["data_dir"], config, self.dataset_id, self.query_mask)
        self.frame.to_csv(self.directory / "experiment_manifest.csv", index=False)
        print(f"{config['mode'].upper()} protocol: {self.protocol['selected_train_images']} images, "
              f"{len(self.selected)} training products; {len(self.folds)} outer folds.", flush=True)

    def pairs(self, directory, filename, products, cap, seed):
        path = Path(directory) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            checksum = path.with_suffix(".sha256")
            if not checksum.exists() or checksum.read_text().strip() != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Saved pair manifest is missing its checksum or has changed: " + str(path))
            pairs = pd.read_csv(path)
        else:
            pairs = make_pairs(self.frame, self.hard_features, products, cap,
                               self.config["negative_candidates"], seed)
            pairs.to_csv(path, index=False)
            path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest(), encoding="ascii")
        allowed = set(products)
        if not set(pairs.query_product_id) <= allowed or not set(pairs.reference_product_id) <= allowed:
            raise ValueError("Pair manifest crosses its product partition.")
        q, r = pairs.query_row.to_numpy(), pairs.reference_row.to_numpy()
        if not np.array_equal(self.frame.image_id.to_numpy()[q], pairs.query_image_id.to_numpy()) or not np.array_equal(
                self.frame.image_id.to_numpy()[r], pairs.reference_image_id.to_numpy()):
            raise ValueError("Pair manifest does not match the current image mapping.")
        expected = (self.frame.class_id.to_numpy()[q] == self.frame.class_id.to_numpy()[r]).astype(int)
        if not np.array_equal(expected, pairs.label.to_numpy()):
            raise ValueError("Pair labels do not match product IDs.")
        return pairs

    def context(self, fold):
        config = self.config
        if fold == "official":
            if not config['run_official_test']:
                raise ValueError('Official holdout is disabled in this configuration.')
            train_products = self.selected
            test_products = sorted(int(p) for p in self.frame.loc[self.frame.source_split.eq("test"), "class_id"].unique())
            directory = self.directory / "official_holdout"
            train_seed, test_seed, inner_seed = config["seed"] + 50001, config["seed"] + 50002, config["seed"] + 51000
        else:
            test_products = self.folds[fold - 1]
            train_products = sorted(set(self.selected) - set(test_products))
            directory = self.directory / f"outer_{fold:02d}"
            train_seed, test_seed, inner_seed = (config["seed"] + fold * 1000 + x for x in (1, 2, 100))
        if set(train_products) & set(test_products):
            raise ValueError("Training and test products overlap.")
        outer_training_products = train_products
        train_products, calibration_products = split_calibration_products(
            self.frame, outer_training_products, config['calibration_fraction'], config['inner_folds'], inner_seed + 900)
        write_json(directory / 'training_partitions.json', {'outer_training_products': outer_training_products,
                   'fitting_products': train_products, 'calibration_products': calibration_products,
                   'test_products': test_products})
        train_pairs = self.pairs(directory, "train_pairs.csv", train_products, config["train_anchor_cap"], train_seed)
        calibration_pairs = self.pairs(directory, 'calibration_pairs.csv', calibration_products,
                                      config['evaluation_anchor_cap'], inner_seed + 901)
        test_pairs = self.pairs(directory, "test_pairs.csv", test_products, config["evaluation_anchor_cap"], test_seed)
        if not self.query_mask[test_pairs.query_row.to_numpy()].all():
            raise ValueError('Test query images are missing from the corruption coverage mask.')
        inner_folds = product_folds(self.frame, train_products, config["inner_folds"], inner_seed)
        write_json(directory / "inner_product_folds.json", inner_folds)
        inner = []
        for index, validation_products in enumerate(inner_folds):
            training_products = sorted(set(train_products) - set(validation_products))
            training = self.pairs(directory, f"inner{index + 1}_train_pairs.csv", training_products,
                                  config["train_anchor_cap"], inner_seed + index * 11 + 1)
            validation = self.pairs(directory, f"inner{index + 1}_validation_pairs.csv", validation_products,
                                    config["evaluation_anchor_cap"], inner_seed + index * 11 + 2)
            inner.append((training_products, validation_products, training, validation))
        return {"fold": fold, "train_products": train_products, "test_products": test_products,
                "calibration_products": calibration_products, "calibration_pairs": calibration_pairs,
                "outer_training_products": outer_training_products,
                "train_pairs": train_pairs, "test_pairs": test_pairs, "inner": inner,
                "seed": inner_seed, "directory": directory}

    def result_directory(self, algorithm):
        path = Path(self.config["results_dir"]) / self.config["run_name"] / self.config["feature_profile"] / algorithm
        data_root = Path(self.config["data_dir"])
        if path == data_root or path.is_relative_to(data_root):
            raise ValueError("Keep experiment results outside the processed image directory.")
        import platform
        import sklearn
        import PIL
        from .sop_feature_bank import _INSTANCES
        files = [Path(__file__).with_name(name) for name in
                 ("sop_baselines.py", "sop_supcon.py", "sop_reporting.py", "sop_config.py",
                  "sop_feature_bank.py", "sop_features.py", "sop_protocol.py", "sop_experiment.py", "sop_calibration.py", "sop_tuning.py")]
        identity = {"shared_protocol_id": self.protocol_id, "config": self.config,
                    "code_fingerprint": file_digest(files), "algorithm_family": algorithm,
                    "python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "sklearn": sklearn.__version__, "Pillow": PIL.__version__,
                    "extractors": [value.metadata for key, value in _INSTANCES.items()
                                   if key[0] in {"hsv_hog_content"} | {c['extractor'] for s in self.config['feature_sets'].values()
                                       for c in s.get('components', [s])}
                                   and key[1] == json.dumps(self.config["extraction"], sort_keys=True)]}
        if algorithm == "supcon":
            import torch
            identity["torch"] = torch.__version__
        guard_directory(path, identity)
        return path
