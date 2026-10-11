"""Run traditional SOP baselines from PyCharm or a terminal.

Run 01_SOP_Preprocessing.py first, then run this script in Unimelb_SML_A2.
Keep sop_features.py and sop_baselines.py in the same directory.

Experiment: Logistic Regression and RBF SVM, each with HSV48 and HSV_HOG192.
Evaluation: manual 10-fold outer / 3-fold inner product-grouped validation,
manual C tuning, validation-derived thresholds and manually computed metrics.
Conditions: clean, partial occlusion, Gaussian noise, blur and rotation.
Plots are saved to files; no Notebook, IPython or interactive display is needed.

This is a traditional-feature pilot. SupCon and ResNet experiments are separate.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

import matplotlib

# Use file-based plots even when PyCharm has no interactive plotting backend.
matplotlib.use("Agg")

import pandas as pd

from sop_baselines import DEFAULT_CONFIG, load_preprocessed, run_experiment


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "processed_sop"
OUTPUT_DIR = SCRIPT_DIR / "results_traditional_py_v1"
CONFIG = copy.deepcopy(DEFAULT_CONFIG)

# Optional settings; change the output directory when changing the protocol.
# CONFIG["use_all_official_train"] = True
# CONFIG["run_official_test"] = False
# CONFIG["c_grids"]["LR"] = [0.001, 0.01, 0.1]


def print_result_tables(output_dir):
    """Print the report tables and saved figure locations after the run."""
    summary = pd.read_csv(output_dir / "summary_mean_sd.csv")
    main = summary[summary["metric"].isin(["FPR", "TPR", "F1", "ROC_AUC"])].copy()
    main["mean +/- SD"] = main.apply(
        lambda row: f"{row['mean']:.4f} +/- {row['sd_across_outer_folds']:.4f}", axis=1
    )
    table = main.pivot(
        index=["algorithm", "features", "negative_subset", "condition"],
        columns="metric", values="mean +/- SD"
    )
    print("\nOuter cross-validation results (mean +/- one fold SD):")
    print(table.to_string())

    audit = json.loads((output_dir / "grid_requirement_audit.json").read_text(encoding="utf-8"))
    print("\nHyperparameter grid requirement audit:")
    print(pd.DataFrame(audit).to_string(index=False))
    print("\nPaired effect of adding spatial gradient features:")
    print(pd.read_csv(output_dir / "paired_delta_mean_sd.csv").to_string(index=False))
    print("For FPR, a negative difference means fewer false matches.")
    print("For TPR, a negative difference means worse same-product matching.")

    official_path = output_dir / "official_holdout" / "metrics.csv"
    if official_path.exists():
        print("\nSupplementary official holdout (not the source of CV error bars):")
        print(pd.read_csv(official_path).to_string(index=False))

    print("\nSaved plots:")
    for filename in ["random_robustness.png", "colour_hard_robustness.png"]:
        print(output_dir / "figures" / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Completed preprocessing directory (default: beside this script).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Experiment directory. Use a new directory for changed settings.")
    parser.add_argument("--all-train", action="store_true",
                        help="Select all eligible official training products.")
    parser.add_argument("--skip-official-test", action="store_true",
                        help="Run nested outer CV without the additional official holdout.")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = copy.deepcopy(CONFIG)
    if args.all_train:
        config["use_all_official_train"] = True
    if args.skip_official_test:
        config["run_official_test"] = False

    print("Python:", sys.executable)
    print("Script directory:", SCRIPT_DIR)
    print("Input directory:", data_dir)
    print("Output directory:", output_dir)
    print("Configuration:")
    print(json.dumps(config, indent=2, ensure_ascii=True))

    manifest, features = load_preprocessed(data_dir)
    print("\nEligible images:", len(manifest))
    print("Products:", manifest["class_id"].nunique())
    print("Image feature matrix:", features.shape)
    print(manifest.groupby("source_split").agg(
        images=("image_id", "size"), products=("class_id", "nunique")
    ).to_string())
    del manifest, features

    run_experiment(data_dir, output_dir, config)
    print_result_tables(output_dir)
    print("\nFinished. See Readme.txt for interpretation and remaining assignment tasks.")


if __name__ == "__main__":
    main()
