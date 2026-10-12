"""Draw the report figures from archived result CSVs (read-only; never trains or reads data).

Figure 1: ROC-AUC across the five test conditions, one panel per algorithm, one line per
feature, error bars = sample SD across outer folds.
Figure 2: paired per-fold differences (added feature minus baseline) of FPR and TPR in
percentage points, one column per algorithm, with the 0 line and the pre-registered
TPR tolerance of -1 percentage point.

Defaults reproduce the round-1 figures (F1-F4; F4 minus F3). After round 2 is archived,
pass its all_metrics.csv / paired_group_deltas.csv and, for example, --added F4,F4.1,F4.2.

Example:
  python tools/plot_report_figures.py
  python tools/plot_report_figures.py --metrics R2/all_metrics.csv \
      --deltas R2/artifacts/comparisons/paired_group_deltas.csv --added F4,F4.1,F4.2
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs" / "experiments" / "2026-10-12-main-experiments"
CONDITIONS = ["clean", "occlusion25", "noise005", "blur15", "rotation15"]
CONDITION_LABELS = ["Clean", "Occlusion\n25%", "Noise\nσ=0.05", "Blur\nr=1.5", "Rotation\n15°"]
ALGORITHMS = ["LR", "SVM", "SupCon"]
ALGORITHM_LABELS = {"LR": "Logistic regression", "SVM": "RBF SVM", "SupCon": "SupCon"}

# Fixed identity per feature (validated categorical palette, fixed order, never cycled).
# Marker and line style give a second encoding for print and colour-vision deficiency.
FEATURES = {
    "HSV48": ("F1 HSV (48-d)", "#2a78d6", "o", "--"),
    "HSV_HOG192": ("F2 HSV + gradient (192-d)", "#eb6834", "s", "--"),
    "ResNet18_Global512": ("F3 ResNet18 global (512-d)", "#1baf7a", "^", "-"),
    "ResNet18_GlobalLocal2560": ("F4 global + 2×2 regions (2560-d)", "#eda100", "D", "-"),
    "ResNet18_GlobalRegionMax1024": ("F4.1 global + region max (1024-d)", "#e87ba4", "v", "-"),
    "ResNet18_GlobalCropMax1024": ("F4.2 global + crop max (1024-d)", "#008300", "P", "-"),
    "ContentHSV48": ("F5 content-only HSV (48-d)", "#6250d6", "o", ":"),
    "ContentHSV_HOG192": ("F5 content-only HSV + gradient (192-d)", "#e34948", "s", ":"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SUBSET_LABELS = {"all": "all negatives", "random": "random negatives", "colour_hard": "colour-similar negatives"}


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def group_id(name):
    return name.split("_")[0]


def save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print("Saved", out / f"{stem}.pdf", "and .png")


def figure_auc(metric_files, groups, subset, out):
    frames = [pd.read_csv(path) for path in metric_files]
    data = pd.concat(frames, ignore_index=True)
    data = data[(data.metric == "ROC_AUC") & (data.negative_subset == subset)]
    data = data[data.group.map(group_id).isin(groups)]
    if data.empty:
        raise SystemExit("No ROC_AUC rows for the selected groups and subset.")
    features = [name for name in FEATURES if name in set(data.features)]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35), sharey=True, constrained_layout=True)
    x = np.arange(len(CONDITIONS))
    offsets = np.linspace(-0.18, 0.18, len(features)) if len(features) > 1 else [0.0]
    for axis, algorithm in zip(axes, ALGORITHMS):
        for offset, feature in zip(offsets, features):
            label, colour, marker, line = FEATURES[feature]
            part = data[(data.algorithm == algorithm) & (data.features == feature)].set_index("condition").reindex(CONDITIONS)
            axis.errorbar(x + offset, part["mean"], yerr=part["sd_across_outer_folds"], color=colour, marker=marker,
                          markersize=4, linestyle=line, linewidth=1.2, elinewidth=0.8, capsize=1.5,
                          markeredgecolor="white", markeredgewidth=0.5, label=label)
        axis.set_title(ALGORITHM_LABELS[algorithm], color=INK)
        axis.set_xticks(x, CONDITION_LABELS)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel(f"ROC-AUC ({SUBSET_LABELS[subset]})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)), frameon=False)
    save(fig, out, f"fig1_auc_by_condition_{subset}")


def paired(deltas, baseline, added):
    """Rows for added minus baseline, whichever order the comparison file stored them in."""
    b, a = deltas.baseline_group.map(group_id), deltas.added_group.map(group_id)
    forward = deltas[(b == baseline) & (a == added)].copy()
    backward = deltas[(b == added) & (a == baseline)].copy()
    backward["delta_added_minus_baseline"] *= -1
    backward["added_features"] = backward["baseline_features"]
    return pd.concat([forward, backward], ignore_index=True)


def figure_deltas(delta_file, baseline, added_groups, subset, out, tolerance_pp=-1.0):
    deltas = pd.read_csv(delta_file)
    deltas = deltas[(deltas.negative_subset == subset) & deltas.metric.isin(["FPR", "TPR"])]
    series = []
    for added in added_groups:
        rows = paired(deltas, baseline, added)
        for feature, part in rows.groupby("added_features"):
            series.append((feature, part))
    if not series:
        raise SystemExit("No paired rows for the selected comparison.")
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.6), sharex=True, sharey="row", constrained_layout=True)
    x = np.arange(len(CONDITIONS))
    width = 0.6 / len(series)
    for column, algorithm in enumerate(ALGORITHMS):
        for row, metric in enumerate(["FPR", "TPR"]):
            axis = axes[row, column]
            axis.axhline(0, color=MUTED, linewidth=0.8)
            if metric == "TPR":
                axis.axhline(tolerance_pp, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
            for index, (feature, part) in enumerate(series):
                label, colour, marker, _ = FEATURES[feature]
                offset = (index - (len(series) - 1) / 2) * width
                cut = part[(part.algorithm == algorithm) & (part.metric == metric)]
                for position, condition in enumerate(CONDITIONS):
                    values = cut[cut.condition == condition].delta_added_minus_baseline.to_numpy() * 100
                    if len(values) == 0:
                        continue
                    jitter = np.linspace(-width * 0.25, width * 0.25, len(values))
                    axis.scatter(position + offset + jitter, values, s=5, color=colour, alpha=0.35, linewidths=0)
                    axis.errorbar(position + offset, values.mean(), yerr=values.std(ddof=1), color=colour,
                                  marker=marker, markersize=4.5, elinewidth=1.0, capsize=1.5,
                                  markeredgecolor="white", markeredgewidth=0.5,
                                  label=f"{label} − {baseline}" if (position == 0 and row == 0 and column == 0) else None)
            axis.grid(axis="y", color=GRID, linewidth=0.6)
            axis.set_axisbelow(True)
            if row == 0:
                axis.set_title(ALGORITHM_LABELS[algorithm], color=INK)
            if column == 0:
                axis.set_ylabel(f"Δ{metric} (pp)")
        axes[1, column].set_xticks(x, CONDITION_LABELS)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3))))
    labels.append(f"TPR tolerance ({tolerance_pp:g} pp)".replace("-", "−"))
    fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(f"Paired differences per outer fold ({SUBSET_LABELS[subset]}); dots = folds, bars = mean ± SD",
                 fontsize=7.5, color=MUTED)
    tag = "_".join(a.replace(".", "-") for a in added_groups)
    save(fig, out, f"fig2_paired_deltas_{tag}_vs_{baseline}_{subset}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, action="append",
                        help="all_metrics.csv (repeatable). Default: round-1 archive.")
    parser.add_argument("--deltas", type=Path, default=R1 / "artifacts" / "comparisons" / "paired_group_deltas.csv")
    parser.add_argument("--groups", default="F1,F2,F3,F4", help="Groups drawn in figure 1.")
    parser.add_argument("--baseline", default="F3")
    parser.add_argument("--added", default="F4", help="Comma-separated groups compared with the baseline.")
    parser.add_argument("--subset", default="all", choices=sorted(SUBSET_LABELS))
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "report-figures")
    args = parser.parse_args()
    style()
    figure_auc(args.metrics or [R1 / "all_metrics.csv"], args.groups.split(","), args.subset, args.out)
    figure_deltas(args.deltas, args.baseline, args.added.split(","), args.subset, args.out)


if __name__ == "__main__":
    main()
