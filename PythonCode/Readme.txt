COMPACT REPORT FIGURE (2026-10-15)
Run: python 04_SOP_Report_Figure.py
Input: the ordinary results/ folder beside 04; ZIP input is not supported.
Uses saved summaries/protocols only; no fitting, model loading or metric recomputation.
Defaults: all eight report groups, LR/SVM/SupCon, five conditions, mixed negatives,
original 95% calibration target, ten outer folds. Handcrafted groups come from
Main_Experiments; CNN groups come from ResNet_LocalFeature_Comparison.
Output: report_figures/figure1_auc_errorbars.png (16 x 7.2 cm, 400 dpi).
Error bars show sample SD across outer folds, not confidence intervals.
Full condition labels (Clean/Occlusion/Noise/Blur/Rotation) are tilted 40 degrees.
Fonts, markers and error-bar lines are enlarged for readability.
Options: --source results --subset all|random|colour_hard. Existing output PNGs
are overwritten by default; --overwrite is optional for older commands.
Other subsets have separate output filenames.
Dependencies: existing matplotlib plus the Python standard library.
No changes to 01-03, experiment settings, or saved experiment results.

Padding groups split (2026-10-14)
Replace 02_SOP_Experiments.py, remove experiment_configs/F5_Traditional_Content_HSV_Gradient.json,
and add F1.1_Traditional_Content_HSV48.json and F2.1_Traditional_Content_HSV_Gradient192.json.
Run: python 02_SOP_Experiments.py --group F1.1,F2.1 --method all
F1.1 and F2.1 are the actual group and output names; no input aliases are used.
The default run name remains Main_Experiments. No changes to 01, 03, config.json or utils.
Move the old results/Main_Experiments/F5_Traditional_Content_HSV_Gradient folder outside results.
After the new runs finish, rename the old results_recall90 folder, then run:
python 03_SOP_Recalibrate_90.py --source results --output results_recall90
03 processes all completed source runs, without fitting models or overwriting existing output.
Historical F5 archives remain valid historical evidence. Eight current research feature groups
exclude the retained F6 pilot; repeated F3/F4 runs remain separate result directories.

90% RECALL RECALIBRATION (no training)
Keep the original 95% experiment results and config unchanged. Run:
python 03_SOP_Recalibrate_90.py
By default it reads results.zip beside the script, otherwise results/.
New files go only to results_recall90/. If that path exists, it stops without
writing. For a new repeat, specify another unused --output directory.
Explicit paths:
python 03_SOP_Recalibrate_90.py --source results.zip --output results_recall90
The new output preserves source run/group/family names. Each family includes
outer_fold_metrics.csv, summary_mean_sd.csv, figures/, comparison_95_vs_90.csv,
recalibrated predictions and calibration records. Source models/tuning stay
in the original location. No pretrained weights, model loading or fitting.
90% means calibration recall target, not accuracy or guaranteed test recall.
The script recalibrates all completed groups in the source. Use results only
for threshold sensitivity analysis; keep independent holdout for final review.

Optional: run python 00_Check_Environment.py first for a read-only environment check.
It checks all groups by default; use --group F1,F2 (same syntax as 02) to check
only the planned groups, for example whether ResNet groups have torch available.
Shared .run configurations select F1 by default; edit --group to select another group.

SOP EXPERIMENTS: TWO ENTRY POINTS

LOCAL GROUPS A/B - 2026-10-12
Plan A uses the single F4.1 component region_max implementation.
The duplicate F7 configuration is removed; former F8 is now F4.2.
F6 remains ResNet50 global2048. F4.1 implements global512 plus channelwise max
of four regional means512; F4.2 implements global512 plus max of five separately
encoded crop descriptors512. Each is1024-dimensional and uses the same ResNet18
IMAGENET1K_V1 weights. F4.1 reuses verified current-version resnet18 caches;
F4.2 requires new crop extraction. Both use all existing corruptions and models.
Do not add F4.1/F4.2 to Main_Experiments: that run name is round R1 (closed),
and automatic comparisons skip old/new source fingerprints. Round R2 reruns the
baselines with both local plans under one new run name (docs/versioning.md):
python 02_SOP_Experiments.py --group F3,F4,F4.1,F4.2 --method all --run-name R2_LocalFeatureAB
(Local_Pilot was an earlier illustrative name, not a recorded round.)
Reusing an existing group's old-version result folder fails its identity check;
old cache fingerprints also differ. Never bypass these compatibility guards.
01 does not need to run again. A uses the existing component transform; B adds a crop extractor,
registry entry, group JSON and preflight weight metadata.
Real pretrained-weight execution and formal performance still need verification
on the GPU computer. Region pooling/cropping is not semantic-part detection.

Use the Unimelb_SML_A2 interpreter in PyCharm.
Run 01_SOP_Preprocessing.py only if preprocessing has not completed.
Run 02_SOP_Experiments.py and select groups, models and a run name in the console.
Model menu: 1 = LR/SVM, 2 = SupCon, 3 = all. Groups accept 1, 1,2, F1,F2 or all.

Script 01 only changed its utility import paths. Existing processed_sop images
can be reused without rerunning preprocessing. All tools are in utils/.
Do not run utility files separately. Retired launchers were removed.
tests/ contains developer checks; it is not part of the experiment run sequence.

CLI commands from this directory:
python 02_SOP_Experiments.py --list
python 02_SOP_Experiments.py --group F1,F2 --method baselines
python 02_SOP_Experiments.py --group F3,F4 --method baselines
python 02_SOP_Experiments.py --group F2,F5 --method baselines   (padding comparison)
python 02_SOP_Experiments.py --group all --method supcon
python 02_SOP_Experiments.py --group F1 --method all --run-name HSV48_repeat1
python 02_SOP_Experiments.py --compare-only

Round R2: local feature plans A (F4.1) and B (F4.2) with the F3/F4 baselines.
F4.1 = F3 global 512 plus the per-channel maximum across the four F4 region
vectors; F4.2 = the same global 512 plus five crops encoded separately and
max-pooled; both 1024. The source fingerprint changed, so F3/F4 rerun in R2:
python 02_SOP_Experiments.py --group F4.1,F4.2 --method all --smoke-test --skip-official-test
python 02_SOP_Experiments.py --group F3,F4,F4.1,F4.2 --method all --run-name R2_LocalFeatureAB
Naming: new method = new number (F7...); improvement of a method = F4.1, F4.2...
End of run: 02 prints "All requested runs finished. Results: <run folder>" after
every requested group and comparison is saved; it is then safe to stop the process.
It also writes exit_watchdog_traceback.txt in the run folder; a normal exit removes
it. If the process has not exited 120 s later, all thread stacks are saved there
to diagnose the hang; results are unaffected.
Formal rounds are numbered R1, R2...; run names use R<round>_<Topic> (no dots).
Round 1 (F1-F4) is the archived Main_Experiments run.

ResNet50 pilot (F6 versus F3, kept but not the current priority). F6 uses frozen ResNet50 IMAGENET1K_V2 weights
(about 98 MB, downloaded by torchvision on first use) with the same padded 224
input as F3. Adding the ResNet50 extractor changed the model-source fingerprint,
so new runs cannot be paired with the 2026-10-12 F1-F4 results; rerun F3 with F6
under a new run name (shared protocol and test pairs are unchanged):
python 02_SOP_Experiments.py --group F6 --method all --smoke-test --skip-official-test
python 02_SOP_Experiments.py --group F3,F6 --method all --run-name R3_ResNet50Pilot

SupCon/ResNet runs initialize PyTorch before experiment setup or reporting.
Runs containing ResNet also check torchvision. A small tensor calculation checks
each requested device, and startup prints the Python executable and device.
An initialization failure stops before any group runs. Handcrafted-only baseline
runs, --list and --compare-only do not require this startup check.
This launcher-only repair does not change model/protocol fingerprints, so completed
folds remain reusable when configuration, data and recorded software versions match.

Code can be copied to a separate GPU computer for execution; this development
computer does not need to run the formal experiments. Returned F1-F4 LR/SVM/SupCon
ten-fold evidence is published under docs/experiments/2026-10-12-main-experiments at
the repository root. Its portable copy excludes trained models and replaces
machine-specific paths; it is an evidence archive, not a resumable run folder.

config.json contains common paths, folds, models and test conditions.
experiment_configs/*.json contain feature groups. A new valid JSON appears automatically.
See experiment_configs/README.md for combinations and new extractors.

Results: results/<run_name>/<group>/{baselines,supcon}/.
Each family saves protocol.json with the fully resolved configuration, source
fingerprint, software versions and extractor metadata. Metrics, predictions,
tuning records, models and robustness figures are saved. Combined model tables
appear after both families finish. comparisons/ reads completed compatible
groups to produce summaries and paired differences, without retraining.
Incompatible comparisons are recorded in a skipped JSON file.

An identical rerun resumes completed folds. --run-name records a fresh rerun.
Same seeds are not independent statistical repeats. Preserve source and environment
as well as saved settings. Changing protocol rules requires a new protocol_dir
shared by all compared groups. Moving utility code changed source fingerprints;
the reorganized default uses a new protocol path. Old data/results remain intact.

Formal defaults: >=10000 official-training images; manual 10 outer/3 inner folds
split by product; separate calibration products; clean training; query-only
occlusion, noise, blur and rotation. Official test is disabled by default.
ResNet uses frozen ImageNet weights and padded 224 images without center crop.
SupCon trains an MLP on selected features; it does not fine-tune ResNet.
--smoke-test is a reduced execution check, never formal experimental evidence.

Dependencies: numpy pandas Pillow OpenCV scikit-learn matplotlib; additionally
torch/torchvision for ResNet/SupCon. First ResNet use downloads model weights.
requirements-base.txt lists base dependencies. Avoid duplicate OpenCV builds.
