Optional: run python 00_Check_Environment.py first for a read-only environment check.
It checks all groups by default; use --group F1,F2 (same syntax as 02) to check
only the planned groups, for example whether ResNet groups have torch available.
Shared .run configurations select F1 by default; edit --group to select another group.

SOP EXPERIMENTS: TWO ENTRY POINTS

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

ResNet50 pilot (F6 versus F3). F6 uses frozen ResNet50 IMAGENET1K_V2 weights
(about 98 MB, downloaded by torchvision on first use) with the same padded 224
input as F3. Adding the ResNet50 extractor changed the model-source fingerprint,
so new runs cannot be paired with the 2026-10-12 F1-F4 results; rerun F3 with F6
under a new run name (shared protocol and test pairs are unchanged):
python 02_SOP_Experiments.py --group F6 --method all --smoke-test --skip-official-test
python 02_SOP_Experiments.py --group F3,F6 --method all --run-name ResNet50_Pilot

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
