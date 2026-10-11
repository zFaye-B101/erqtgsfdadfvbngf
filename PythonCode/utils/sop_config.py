"""Load a saved experiment configuration for the shared numbered entry point.

Groups inherit common settings; paths are relative to the common JSON file.
"""

import copy
import json
import math
from pathlib import Path


def discover_experiments(directory):
    """Discover arbitrary groups without a hardcoded menu."""
    groups = {}
    for path in sorted(Path(directory).glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        name = data['group']
        if name in groups:
            raise ValueError('Duplicate experiment group: ' + name)
        groups[name] = (path, data)
    return groups


def load_config(path, smoke=False, experiment_path=None):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for field in ("data_dir", "protocol_dir", "cache_dir", "results_dir"):
        value = Path(config[field]).expanduser()
        config[field] = str((value if value.is_absolute() else path.parent / value).resolve())
    if experiment_path is not None:
        profile = json.loads(Path(experiment_path).read_text(encoding='utf-8'))
        config['feature_profile'] = profile['group']
    else:
        profile = config["feature_profiles"][config["feature_profile"]]
    config["feature_sets"] = copy.deepcopy(profile["features"])
    config["comparisons"] = copy.deepcopy(profile.get("comparisons", []))
    config.pop("feature_profiles", None)
    config.pop('experiment_dir', None)
    config.update(copy.deepcopy(config["baselines"]))
    if smoke:
        config.update(mode="smoke", minimum_train_images=384, outer_folds=2,
                      inner_folds=2, train_anchor_cap=24, evaluation_anchor_cap=12,
                      use_all_official_train=False, run_official_test=False,
                      maximum_grid_extensions=0)
        config["supcon"].update(epochs=1, products_per_batch=8, maximum_grid_extensions=0)
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
    if not 0 < config['calibration_fraction'] < .5:
        raise ValueError('Calibration fraction must be in (0, 0.5).')
    for field in ("train_anchor_cap", "evaluation_anchor_cap", "negative_candidates"):
        if config[field] < 1:
            raise ValueError(field + " must be positive.")
    for name, spec in config["feature_sets"].items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Feature names must contain letters, numbers or underscores.")
        for component in spec.get('components', [spec]):
            start, end = component['columns']
            if start < 0 or end <= start:
                raise ValueError('Invalid feature column range: ' + name)
            if component.get('normalization', 'none') not in {'none', 'l2'}:
                raise ValueError('Component normalization must be none or l2.')
    for base, added in config["comparisons"]:
        if base not in config["feature_sets"] or added not in config["feature_sets"]:
            raise ValueError("Comparison references an unselected feature.")
    for grid in config["c_grids"].values():
        if len(grid) != 3 or not 0 < grid[0] < grid[1] < grid[2]:
            raise ValueError("Each classifier requires three increasing positive C values.")
    sup = config["supcon"]
    grid = sup["temperature_grid"]
    if len(grid) != 3 or not 0 < grid[0] < grid[1] < grid[2] or any(not math.isfinite(t) for t in grid):
        raise ValueError("SupCon needs three increasing finite positive temperatures.")
    lower, upper = sup['temperature_bounds']
    if not 0 < lower <= grid[0] < grid[2] <= upper or not math.isfinite(upper):
        raise ValueError('Temperature bounds must contain the initial positive grid.')
    if sup['maximum_grid_extensions'] < 0 or not math.isfinite(sup['temperature_extension_factor']) or sup['temperature_extension_factor'] <= 1:
        raise ValueError('Temperature extensions must be nonnegative and the factor finite and >1.')
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
