SOP MATCHING EXPERIMENT PIPELINE

Fixed entry points
00_Check_Environment.py: read-only preflight. Checks the interpreter, packages,
PyTorch/CUDA, configuration, data or ZIP location, free disk space, write access
and, on Windows, path length and console encoding. Installs and downloads nothing.
Exit code 1 means at least one FAIL item must be fixed first.
01_SOP_Preprocessing.py: audits the original ZIP images, excludes unreadable/missing
images, originals with either side below 100 pixels, and products with only
one eligible image. It exports padded 224x224 RGB JPEGs and audit tables while
preserving official train/test labels. Run once per dataset.
02_SOP_Experiments.py: runs nested product-fold LR/SVM baselines or SupCon.

From the PythonCode directory, using your activated Python 3.11 environment:
python 00_Check_Environment.py
python 01_SOP_Preprocessing.py
python 02_SOP_Experiments.py --method baselines
python 02_SOP_Experiments.py --method supcon

01 searches beside the script and its parent directories for Stanford_Online_Products.zip.
An explicit input is supported:
python 01_SOP_Preprocessing.py --zip-path "D:/data/Stanford_Online_Products.zip"
Nonempty preprocessing outputs are never overwritten. If the original 01
already finished and clean_manifest.csv, audit_summary.json and images/ exist,
set data_dir to that folder and run either experiment method directly. Old feature NPZs are ignored;
the new feature bank builds reusable caches on the first experiment run.

Windows and PyCharm
Open the repository root (the folder containing .run and PythonCode) as the
PyCharm project, then select the Python 3.11 interpreter that has the packages
above: Settings > Project > Python Interpreter. Shared run configurations in
.run appear in the Run widget:
  00 Check environment       python 00_Check_Environment.py
  01 Preprocess SOP          python 01_SOP_Preprocessing.py
  02 Smoke baselines         --method baselines --smoke-test --skip-official-test
  02 Smoke SupCon            --method supcon --smoke-test --skip-official-test
  02 Baselines / 02 SupCon   formal runs with experiment_config.json
  Tests PythonCode           python -m unittest discover -s tests -v
All use PythonCode as the working directory and set PYTHONUNBUFFERED=1 and
PYTHONUTF8=1. They use the project interpreter; if PyCharm reports a missing
module or interpreter (for example after renaming the project folder), choose
the interpreter in Run > Edit Configurations.
Recommended order on a new machine: 00, then 01 (once; place the ZIP beside the
repository or add --zip-path in the 01 configuration), then the two smoke runs,
and only then the formal runs. Keep the project in a short folder such as
D:\sml\ because nested result paths are long; 00 warns when the estimated
path length exceeds the Windows limit without long-path support. A CPU-only
PyTorch build cannot use an NVIDIA GPU; 00 reports this case.

Dependencies
Use the same Python interpreter for installation and IDE execution.
Base libraries: numpy, pandas, Pillow, OpenCV, scikit-learn and matplotlib.
requirements-base.txt provides an optional pip installation list. Existing
Conda OpenCV also works; do not install another OpenCV build unnecessarily.
SupCon and ResNet features additionally need compatible torch/torchvision.
For CPU:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
Device settings accept auto, cpu, cuda or mps. Auto selects an available
accelerator then CPU. CPU-only PyTorch cannot use an NVIDIA GPU.

Changing features
Edit feature_profile in experiment_config.json; no preprocessing or experiment code changes are needed:
handcrafted: original padded HSV48 versus HSV_HOG192 (retained baseline).
handcrafted_content: ContentHSV48 versus ContentHSV_HOG192, excluding padding.
handcrafted_padding_audit: all four padded/content variants, paired comparisons
of padding removal for each dimension. Use this profile for development checks.
The content rectangle is recovered from original dimensions, not detected from
pixel colour. HSV counts and HOG cells operate only on that rectangle, after
the same JPEG decode and query corruption. HOG resizes the rectangle to its
128x128 descriptor grid. ResNet still uses its documented padded input; this
update does not claim padding has been removed from CNN features.
resnet18_global: frozen ImageNet ResNet18 global pooling, 512 dimensions.
resnet18: global512 versus global512 plus four 2x2 pooled region descriptors,
2560 dimensions total. This paired comparison measures added spatial detail.
The ResNet classifier is removed; IMAGENET1K_V1 weights are frozen and downloaded
by torchvision on first use. ImageNet normalization is applied to the same
padded 224x224 images used by other features, without an extra center crop.

Example configuration change:
"feature_profile": "resnet18_global"
Rerun 02 with each desired --method. No preprocessing rerun is needed. Outputs are separated
by profile; both methods reuse the same extracted features and corruption caches.

For a new fixed extractor, implement dimension, metadata and __call__(images)
returning an N x D float array in sop_feature_bank.py. Register it in EXTRACTORS
and add a configuration profile with extractor and column ranges. With one
feature use comparisons: []; with variants declare baseline/added comparisons.
An extractor learned from SOP labels must be fitted inside training folds,
not globally in the feature bank, to avoid validation/test leakage.

Shared experiment protocol
Formal defaults select whole official-training products across categories
until at least 10,000 cleaned original images are included. All images of those
products are retained. use_all_official_train can include every eligible train
product. Ten outer and three inner folds split PRODUCTS, with manual category
stratification. Official test results are supplementary and are never used
for fitting, tuning or threshold selection. run_official_test defaults to false;
enable it only after the protocol is finalized.

Each sampled anchor has a same-product positive, same-category random negative
and same-category colour-hard negative. Hard selection always uses clean CONTENT-ONLY HSV48,
including ResNet experiments. Saved product folds, pair CSVs and checksums are
shared across algorithms and feature profiles. Feature changes preserve pairs.
All methods train on clean inputs. Test conditions corrupt only the query;
reference images remain clean. Occlusion/noise/blur/rotation operate on image
content, keeping padding fixed. Features are recomputed from corrupted images.

LR/SVM use the absolute difference only: abs(query - reference), with D pair
dimensions for D image dimensions. No elementwise-product term is concatenated.
Input scaling and C search are fitted within inner training folds; three C
candidates are tested with optional boundary extensions. SupCon searches three
temperatures by clean inner-validation matching AUC. A boundary winner can
extend the three-value window up to maximum_grid_extensions=2 times with a
factor of 2, within temperature_bounds=[0.005, 1.0]. Only new candidates are
trained; evaluated candidates are reused and equal AUC favours the middle.
All decisions use inner validation only. Exhausted boundaries remain flagged. Its scaler and neural
network are independently refitted in every inner and outer training partition.
Architecture and epochs are preset. Outer tests never select settings.
Each outer training set reserves category-stratified calibration products:
calibration_fraction=0.2, rounded up with >=2 products per category. At least
2*inner_folds products per category remain for model fitting and inner tuning.
Calibration products are excluded from all tuning, scaler fitting and model
training. The selected final model is fitted once on fitting products, scored
on clean calibration pairs, and its threshold is chosen for target recall 0.95.
That SAME fixed model evaluates all test conditions; it is not refitted after
calibration. Test recall still need not equal the finite-sample calibration
target. training_partitions.json and calibration.json record these partitions.
The >=10000-image count describes the selected dataset, not every fitting fold.
Small datasets lacking enough products fail clearly instead of reusing products.
Metrics are calculated manually. Report outer-fold means and sample standard
deviations, all/random/colour-hard subsets, and separate official holdout results.

Training budgets differ: LR/SVM use capped sampled pairs; SupCon samples clean
images from every training product across epochs unless train_product_cap is
set. Shared tests make evaluation comparable, but training compute is not equal.
Record budgets and external ImageNet pretraining when interpreting comparisons.
A single 512-dimensional profile does not provide a spatial-feature ablation.

SupCon paper adaptation
Khosla et al., Supervised Contrastive Learning, NeurIPS 2020:
https://arxiv.org/abs/2004.11362
https://github.com/HobbitLong/SupContrast
sop_supcon.py implements Eq. (2): normalized representations, self exclusion,
all same-product positives, mean positive log probabilities outside the log.
Each batch contains >=2 products and >=2 distinct clean images per product;
other products supply negatives.
Frozen descriptors feed a trainable MLP encoder and projection head. Inference
retains the normalized encoder representation and discards the projection
head for cosine scoring. Checkpoints save weights, training-only scaling,
settings and training products. load_model supports trusted local inference.

Differences from the paper: fixed descriptors instead of an end-to-end CNN;
different clean product images instead of two stochastic augmented views of
each image; unseen-product matching instead of linear ImageNet classification.
This is a project adaptation and does not claim the original classification
results. Test conditions do not enable training augmentation. Twenty epochs
is a configurable starting budget, not the original paper training schedule.

Configuration, outputs and resuming
Paths resolve relative to experiment_config.json. --config selects another file.
02 accepts --method baselines|supcon, --data-dir, --results-dir, --protocol-dir,
--all-train, --skip-official-test and --smoke-test. Use identical common settings
for comparisons.
data_dir: reusable images and clean manifest.
cache_dir: extractor-specific feature arrays and metadata.
protocol_dir: common folds, pairs and integrity checks.
results_dir/run_name/feature_profile/baselines or supcon: tuning, per-pair
predictions, checkpoints, fold metrics, summary CSVs and figures.
After both experiment methods finish, combined_summary_mean_sd.csv combines all methods.
paired_feature_deltas.csv and paired_delta_mean_sd.csv contain ablations.
grid_requirement_audit.json flags hyperparameter boundary choices; review these
using inner validation only.

Completed outer folds resume only under identical configuration/data/code.
Changing model settings requires a new run_name. Changing data, sampling,
folds, conditions or protocol code also requires a new protocol_dir. Changed
extractor code receives a new cache identity. Dataset fingerprints cover
manifest bytes plus image file size/mtime, not every image's content hash.
Do not edit saved pair CSVs.
Version 3 defaults use experiment_v3, protocols/shared_v3, feature_cache_v3 and
results_v3. Keep v2 outputs as historical evidence; do not reuse their completed
folds or thresholds. Existing processed_sop images remain compatible, so 01
does not need to be rerun. Switching feature profiles shares the v3 pair lists.

Execution checks
python 00_Check_Environment.py
python 02_SOP_Experiments.py --method baselines --smoke-test
python 02_SOP_Experiments.py --method supcon --smoke-test
These select at least 384 images (more if product counts require it), 2 outer/2 inner folds, smaller pair caps and one SupCon
epoch, writing to separate folders. They are execution checks, not formal
experimental evidence. Full nested runs can take substantial time.
python -m unittest discover -s tests -v
Synthetic checks cover loss, gradients, train-only scaling, checkpoint reload,
512-dimensional input, shared folds/pairs, metrics and resume. The ResNet check
uses random weights for dimensions/frozen inference, not pretrained accuracy.

File layout
00_Check_Environment.py is the read-only preflight check.
01_SOP_Preprocessing.py is the preprocessing command and implementation.
02_SOP_Experiments.py selects LR/SVM or SupCon with --method.
sop_features.py contains image resizing, HSV/HOG and SIFT helpers.
sop_feature_bank.py contains fixed extractors and feature-cache logic.
sop_protocol.py contains product splits, pairs and protocol checks.
sop_experiment.py coordinates shared experiment state.
sop_baselines.py and sop_supcon.py contain the model families.
sop_calibration.py, sop_tuning.py and sop_reporting.py contain separate evaluation utilities.
sop_config.py loads and validates configuration.

Historical notebooks are not active entry points for the new configuration.
