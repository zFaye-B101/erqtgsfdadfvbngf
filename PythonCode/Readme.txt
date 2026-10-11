SOP MATCHING EXPERIMENT PIPELINE

Fixed entry points
01_SOP_Preprocessing.py: audit original ZIP images, exclude unreadable/missing
images, originals with either side below 100 pixels, and products with only
one eligible image. Export padded 224x224 RGB JPEGs and audit tables. Preserve
official train/test labels. Run once per dataset.
02_SOP_Traditional_Baselines.py: nested product-fold LR and RBF SVM experiments.
03_SOP_SupCon.py: nested supervised contrastive representation learning and
cosine matching on unseen products.

From the PythonCode directory, using your activated Python 3.11 environment:
python 01_SOP_Preprocessing.py
python 02_SOP_Traditional_Baselines.py
python 03_SOP_SupCon.py

01 searches beside Code and its parents for Stanford_Online_Products.zip.
An explicit input is supported:
python 01_SOP_Preprocessing.py --zip-path "D:/data/Stanford_Online_Products.zip"
Nonempty preprocessing outputs are never overwritten. If the original 01
already finished and clean_manifest.csv, audit_summary.json and images/ exist,
set data_dir to that folder and run 02/03 directly. Old feature NPZs are ignored;
the new feature bank builds reusable caches on the first experiment run.

Dependencies
Use the same Python interpreter for installation and IDE execution.
Base libraries: numpy, pandas, Pillow, OpenCV, scikit-learn and matplotlib.
requirements-base.txt provides an optional pip installation list. Existing
Conda OpenCV also works; do not install another OpenCV build unnecessarily.
03 and ResNet features additionally need compatible torch/torchvision.
For CPU:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
Device settings accept auto, cpu, cuda or mps. Auto selects an available
accelerator then CPU. CPU-only PyTorch cannot use an NVIDIA GPU.

Changing features
Edit feature_profile in experiment_config.json; keep 01/02/03 unchanged:
handcrafted: HSV48 versus HSV_HOG192.
resnet18_global: frozen ImageNet ResNet18 global pooling, 512 dimensions.
resnet18: global512 versus global512 plus four 2x2 pooled region descriptors,
2560 dimensions total. This paired comparison measures added spatial detail.
The ResNet classifier is removed; IMAGENET1K_V1 weights are frozen and downloaded
by torchvision on first use. ImageNet normalization is applied to the same
padded 224x224 images used by other features, without an extra center crop.

Example configuration change:
"feature_profile": "resnet18_global"
Rerun only 02 and 03. No preprocessing rerun is needed. Outputs are separated
by profile; 02 and 03 reuse the same extracted features and corruption caches.

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
for fitting, tuning or threshold selection.

Each sampled anchor has a same-product positive, same-category random negative
and same-category colour-hard negative. Hard selection always uses clean HSV48,
including ResNet experiments. Saved product folds, pair CSVs and checksums are
shared across algorithms and feature profiles. Feature changes preserve pairs.
All methods train on clean inputs. Test conditions corrupt only the query;
reference images remain clean. Occlusion/noise/blur/rotation operate on image
content, keeping padding fixed. Features are recomputed from corrupted images.

LR/SVM use absolute difference plus elementwise product of image features.
Input scaling and C search are fitted within inner training folds; three C
candidates are tested with optional boundary extensions. SupCon searches three
temperatures by clean inner-validation matching AUC. Its scaler and neural
network are independently refitted in every inner and outer training partition.
Architecture and epochs are preset. Outer tests never select settings.
Thresholds use inner out-of-fold positive scores for the configured validation
recall (0.95). Outer-test recall is not guaranteed to equal that target.
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
02/03 accept --data-dir, --results-dir, --protocol-dir, --all-train and
--skip-official-test. Both must use identical common settings for comparisons.
data_dir: reusable images and clean manifest.
cache_dir: extractor-specific feature arrays and metadata.
protocol_dir: common folds, pairs and integrity checks.
results_dir/run_name/feature_profile/baselines or supcon: tuning, per-pair
predictions, checkpoints, fold metrics, summary CSVs and figures.
After both 02 and 03 finish, combined_summary_mean_sd.csv combines all methods.
paired_feature_deltas.csv and paired_delta_mean_sd.csv contain ablations.
grid_requirement_audit.json flags hyperparameter boundary choices; review these
using inner validation only.

Completed outer folds resume only under identical configuration/data/code.
Changing model settings requires a new run_name. Changing data, sampling,
folds, conditions or protocol code also requires a new protocol_dir. Changed
extractor code receives a new cache identity. Dataset fingerprints cover
manifest bytes plus image file size/mtime, not every image's content hash.
Do not edit saved pair CSVs.

Execution checks
python 02_SOP_Traditional_Baselines.py --smoke-test
python 03_SOP_SupCon.py --smoke-test
These use 384 images, 2 outer/2 inner folds, smaller pair caps and one SupCon
epoch, writing to separate folders. They are execution checks, not formal
experimental evidence. Full nested runs can take substantial time.
python -m unittest discover -s tests -v
Synthetic checks cover loss, gradients, train-only scaling, checkpoint reload,
512-dimensional input, shared folds/pairs, metrics and resume. The ResNet check
uses random weights for dimensions/frozen inference, not pretrained accuracy.

Original Python versions remain available in Git history. Historical notebooks are
not active entry points for the new configuration.
