"""Load the one editable configuration shared by the three fixed entry points."""

import copy
import json
import math
from pathlib import Path


def load_config(path, smoke=False):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for field in ("data_dir", "protocol_dir", "cache_dir", "results_dir"):
        value = Path(config[field]).expanduser()
        config[field] = str((value if value.is_absolute() else path.parent / value).resolve())
    profile = config["feature_profiles"][config["feature_profile"]]
    config["feature_sets"] = copy.deepcopy(profile["features"])
    config["comparisons"] = copy.deepcopy(profile.get("comparisons", []))
    config.pop("feature_profiles")
    config.update(copy.deepcopy(config["baselines"]))
    if smoke:
        config.update(mode="smoke", minimum_train_images=384, outer_folds=2,
                      inner_folds=2, train_anchor_cap=24, evaluation_anchor_cap=12,
                      use_all_official_train=False, run_official_test=False,
                      maximum_grid_extensions=0)
        config["supcon"].update(epochs=1, products_per_batch=8)
        for field in ("protocol_dir", "cache_dir", "results_dir"):
            config[field] += "_smoke"
    validate_config(config)
    return config


def validate_config(config):
    for field in ("run_name", "feature_profile"):
        if not config[field] or not all(c.isalnum() or c in "_-" for c in config[field]):
            raise ValueError(field + " must contain only letters, numbers, underscores or hyphens.")
    formal = config["mode"] == "formal"
    if config["mode"] not in {"formal", "smoke"}:
        raise ValueError("mode must be formal or smoke.")
    if formal and (config["outer_folds"] < 10 or config["inner_folds"] < 3
                   or config["minimum_train_images"] < 10000):
        raise ValueError("Formal runs require >=10000 images, >=10 outer and >=3 inner folds.")
    if min(config["outer_folds"], config["inner_folds"]) < 2:
        raise ValueError("At least two folds are required.")
    if not config["feature_sets"] or list(config["conditions"])[0] != "clean":
        raise ValueError("Select features and put clean first in conditions.")
    if not 0 < config["target_validation_recall"] <= 1:
        raise ValueError("Validation recall must be in (0, 1].")
    for field in ("train_anchor_cap", "evaluation_anchor_cap", "negative_candidates"):
        if config[field] < 1:
            raise ValueError(field + " must be positive.")
    for name, spec in config["feature_sets"].items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Feature names must contain letters, numbers or underscores.")
        start, end = spec["columns"]
        if start < 0 or end <= start:
            raise ValueError("Invalid feature column range: " + name)
    for base, added in config["comparisons"]:
        if base not in config["feature_sets"] or added not in config["feature_sets"]:
            raise ValueError("Comparison references an unselected feature.")
    for grid in config["c_grids"].values():
        if len(grid) != 3 or not 0 < grid[0] < grid[1] < grid[2]:
            raise ValueError("Each classifier requires three increasing positive C values.")
    sup = config["supcon"]
    grid = sup["temperature_grid"]
    if len(grid) < 3 or len(set(grid)) != len(grid) or any(not math.isfinite(t) or t <= 0 for t in grid):
        raise ValueError("SupCon needs at least three distinct positive temperatures.")
    if sup["view_mode"] != "clean_product_images":
        raise ValueError("Implemented SupCon views are distinct clean images of each product.")
    if sup["products_per_batch"] < 2 or sup["images_per_product"] < 2:
        raise ValueError("SupCon needs >=2 products and >=2 images per product in a batch.")
    if min(sup["epochs"], sup["cpu_threads"], sup["hidden_dim"],
           sup["representation_dim"], sup["projection_dim"]) < 1:
        raise ValueError("SupCon sizes, epochs and thread count must be positive.")
    if not math.isfinite(sup["learning_rate"]) or sup["learning_rate"] <= 0 or not math.isfinite(sup["weight_decay"]) or sup["weight_decay"] < 0:
        raise ValueError("SupCon learning rate must be positive and weight decay nonnegative and finite.")
    if config["extraction"]["batch_size"] < 1:
        raise ValueError("Extraction batch size must be positive.")
