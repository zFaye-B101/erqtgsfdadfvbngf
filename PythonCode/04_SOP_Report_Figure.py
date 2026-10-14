"""Draw one compact report figure from saved experiment summaries.

No training, feature extraction, or metric recomputation. Directory input only.
Default input: results/ beside this script; no compressed archive is needed.
Default output: report_figures/figure1_auc_errorbars.png beside this script.
An existing output PNG is overwritten automatically.
Error bars show mean +/- one sample SD across ten outer test folds.
All eight report feature groups are included by default.
Default figure size is 16 x 7.2 cm, with full condition labels tilted 40 degrees.

Usage:
    python 04_SOP_Report_Figure.py
    python 04_SOP_Report_Figure.py --source results --subset random
    python 04_SOP_Report_Figure.py --source results --overwrite
"""

import argparse
import csv
import io
import json
import math
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
GROUPS = {
    "F1": "F1_Traditional_HSV48",
    "F1.1": "F1.1_Traditional_Content_HSV48",
    "F2": "F2_Traditional_HSV_Gradient192",
    "F2.1": "F2.1_Traditional_Content_HSV_Gradient192",
    "F3": "F3_ResNet_Global512",
    "F4": "F4_ResNet_GlobalLocal2560",
    "F4.1": "F4.1_ResNet_GlobalRegionMax1024",
    "F4.2": "F4.2_ResNet_GlobalCropMax1024",
}
ALGORITHMS = ("LR", "SVM", "SupCon")
CONDITIONS = ("clean", "occlusion25", "noise005", "blur15", "rotation15")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#E69F00",
          "#CC79A7", "#332288", "#56B4E9", "#555555")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


class SummarySource:
    """Read only summaries/protocols from an ordinary results directory."""

    def __init__(self, source):
        self.path = source.expanduser().resolve()
        if self.path.is_dir():
            self.names = [p.relative_to(self.path).as_posix()
                          for p in self.path.rglob("summary_mean_sd.csv")]
        else:
            raise FileNotFoundError("Select an ordinary results directory: " + str(self.path))

    def read(self, name):
        path = (self.path / name).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError("Input path escapes the results directory.")
        return path.read_bytes()

    def select(self, run, group, family):
        tail = (run, group, family, "summary_mean_sd.csv")
        matches = [n for n in self.names if PurePosixPath(n).parts[-4:] == tail]
        if len(matches) != 1:
            raise ValueError("Expected exactly one summary for " + "/".join(tail)
                             + "; found " + str(len(matches)))
        name = matches[0]
        protocol = json.loads(self.read(str(PurePosixPath(name).parent / "protocol.json")))
        config = protocol["config"]
        if config["target_validation_recall"] != 0.95 or config["outer_folds"] != 10:
            raise ValueError("This figure requires original 95%-target, ten-fold results.")
        return list(csv.DictReader(io.StringIO(self.read(name).decode("utf-8-sig"))))

def load_points(source, groups, main_run, local_run, subset):
    """Use Main for handcrafted groups and the common local run for CNNs."""
    points = {}
    for label in groups:
        run = main_run if label in {"F1", "F1.1", "F2", "F2.1"} else local_run
        family_rows = {f: source.select(run, GROUPS[label], f)
                       for f in ("baselines", "supcon")}
        for algorithm in ALGORITHMS:
            rows = family_rows["supcon" if algorithm == "SupCon" else "baselines"]
            for condition in CONDITIONS:
                selected = [r for r in rows if r["algorithm"] == algorithm
                            and r["condition"] == condition
                            and r["negative_subset"] == subset
                            and r["metric"] == "ROC_AUC"]
                if len(selected) != 1:
                    raise ValueError("Missing/duplicate " + subset + " AUC: "
                                     + "/".join((label, algorithm, condition)))
                row = selected[0]
                mean, sd = float(row["mean"]), float(row["sd_across_outer_folds"])
                if not (math.isfinite(mean) and math.isfinite(sd)
                        and 0 <= mean <= 1 and sd >= 0
                        and int(row["outer_fold_count"]) == 10
                        and row.get("mode") == "formal"):
                    raise ValueError("Invalid or non-formal summary row.")
                points[label, algorithm, condition] = (mean, sd)
    return points


def draw(points, groups, output, width_cm, height_cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, sharey=True,
                             figsize=(width_cm / 2.54, height_cm / 2.54))
    lows = [m - s for m, s in points.values()]
    highs = [m + s for m, s in points.values()]
    margin = max(0.012, (max(highs) - min(lows)) * 0.07)
    for axis, algorithm in zip(axes, ALGORITHMS):
        for index, label in enumerate(groups):
            values = [points[label, algorithm, c] for c in CONDITIONS]
            offset = (index - (len(groups) - 1) / 2) * 0.07
            axis.errorbar([j + offset for j in range(5)], [v[0] for v in values],
                          yerr=[v[1] for v in values], label=label,
                          color=COLORS[index], marker=MARKERS[index],
                          markersize=4.0, linewidth=1.05, elinewidth=1.0,
                          capsize=2.5, markeredgewidth=0.8,
                          linestyle="--" if label in {"F1", "F2", "F1.1", "F2.1"} else "-")
        axis.set_title(algorithm, pad=7)
        axis.set_xticks(range(5), ("Clean", "Occlusion", "Noise", "Blur", "Rotation"))
        plt.setp(axis.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
        axis.set_ylim(max(0, min(lows) - margin), min(1, max(highs) + margin))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(width=0.7, length=3, pad=3)
    axes[0].set_ylabel("ROC-AUC", labelpad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.54, 0.015),
               ncol=len(groups), frameon=False, columnspacing=0.5,
               handlelength=1.25, handletextpad=0.25, borderaxespad=0)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.89, bottom=0.34, wspace=0.16)
    # Keep exact physical size: do not use bbox_inches='tight'.
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path,
                        help="Optional PNG path; otherwise use a subset-specific filename.")
    parser.add_argument("--groups", default="F1,F1.1,F2,F2.1,F3,F4,F4.1,F4.2")
    parser.add_argument("--subset", choices=("all", "random", "colour_hard"), default="all",
                        help="Negative subset: all=mixed, random, or colour_hard.")
    parser.add_argument("--main-run", default="Main_Experiments")
    parser.add_argument("--local-run", default="ResNet_LocalFeature_Comparison")
    parser.add_argument("--width-cm", type=float, default=16.0)
    parser.add_argument("--height-cm", type=float, default=7.2)
    parser.add_argument("--overwrite", action="store_true",
                        help="Optional compatibility flag; output PNG is overwritten by default.")
    args = parser.parse_args()
    groups = [g.strip() for g in args.groups.split(",")]
    if not groups or len(groups) != len(set(groups)) or any(g not in GROUPS for g in groups):
        parser.error("Select unique known group names, e.g. F1,F1.1,F2,F2.1,F3,F4,F4.1,F4.2.")
    if args.width_cm < 16 or args.height_cm < 7:
        parser.error("Use width >= 16 cm and height >= 7 cm for full condition labels.")
    suffix = "" if args.subset == "all" else "_" + args.subset
    output = (args.output or ROOT / ("report_figures/figure1_auc_errorbars" + suffix + ".png"))
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".png":
        parser.error("Output must be a .png file.")
    source_path = args.source
    if source_path is None:
        source_path = ROOT / "results"
    source = SummarySource(source_path)
    points = load_points(source, groups, args.main_run, args.local_run, args.subset)
    draw(points, groups, output, args.width_cm, args.height_cm)
    print("Saved:", output)
    print("Size:", args.width_cm, "x", args.height_cm, "cm; subset:", args.subset, "; 95% target.")
    print("Error bars: mean +/- one sample SD across ten outer test folds (not a CI).")
    print("Conditions: Clean; Occlusion 25%; Noise sigma 0.05; Blur radius 1.5; Rotation +15 deg.")
    print("No models, predictions or input result files were changed.")


if __name__ == "__main__":
    main()
