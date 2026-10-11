"""Unified CLI for LR/SVM baselines and SupCon experiments."""

import argparse
from pathlib import Path
import sys

from sop_config import load_config
from sop_experiment import Experiment
from sop_reporting import combine_results, print_tables


def main():
    parser = argparse.ArgumentParser(description="Run SOP experiments using the shared configuration.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    parser.add_argument("--method", choices=("baselines", "supcon"), required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--all-train", action="store_true")
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Explicit 2/2-fold, 384-image, 1-epoch check; never formal evidence.")
    args = parser.parse_args()
    family = args.method
    config = load_config(args.config, args.smoke_test)
    for field in ("data_dir", "results_dir", "protocol_dir"):
        override = getattr(args, field)
        if override is not None:
            config[field] = str(override.expanduser().resolve())
    if args.all_train:
        config["use_all_official_train"] = True
    if args.skip_official_test:
        config["run_official_test"] = False
    print("Python:", sys.executable)
    print("Feature profile:", config["feature_profile"])
    print("Run mode:", config["mode"].upper())
    print("Input:", config["data_dir"])
    print("Public evaluation protocol:", config["protocol_dir"])
    if config["mode"] == "smoke":
        print("SMOKE TEST ONLY: reduced folds, images and epochs do not satisfy formal evaluation.")
    if family == "baselines":
        from sop_baselines import run_experiment
    elif family == "supcon":
        from sop_supcon import run_experiment
    else:
        raise ValueError("Unknown experiment family.")
    experiment = Experiment(config)
    directory = run_experiment(experiment)
    print_tables(directory)
    combine_results(experiment)


if __name__ == "__main__":
    main()
