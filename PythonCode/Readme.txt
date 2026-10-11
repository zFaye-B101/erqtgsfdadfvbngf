COMP90051 SOP traditional baseline experiments
=============================================

Files to put together in your PyCharm project root:
  01_SOP_Preprocessing.py
  sop_features.py
  02_SOP_Traditional_Baselines.py
  sop_baselines.py
  Stanford_Online_Products.zip (local only)

Environment: activate Unimelb_SML_A2 and select its Python interpreter in PyCharm.
Packages are already included in the environment previously installed:
numpy, pandas, scipy, scikit-learn, matplotlib, Pillow and opencv.
No Notebook, IPython, torch, GPU or pretrained weights are needed for 02.

1. Run 01_SOP_Preprocessing.py ONCE. It writes processed_sop/ beside the script.
   If processed_sop already contains a completed clean_manifest.csv and
   image_features_hsv_hog.npz plus images/, do not rerun 01 into that folder.
2. Run 02_SOP_Traditional_Baselines.py using the same environment.
   Check DATA_DIR, OUTPUT_DIR and CONFIG near the top if changing defaults.
   Do not run sop_features.py or sop_baselines.py separately.
3. Results go to results_traditional_py_v1/. Keep data, caches and model weights
   local; do not upload or include them in the submission code ZIP.
4. If interrupted, rerun 02 with the IDENTICAL configuration, source code,
   input data and environment. Completed outer folds and caches are reused.
   Changed settings/code require a NEW output directory. No results are mixed.

Terminal commands (run from your project root):
  conda activate Unimelb_SML_A2
  python 01_SOP_Preprocessing.py
  python 02_SOP_Traditional_Baselines.py

In PyCharm, select the environment's python.exe, open a numbered script, and
use Run. Both scripts use their own directory for default paths, rather than
depending on the Run Configuration's working directory.

Optional custom paths:
  python 01_SOP_Preprocessing.py --zip-path "D:/data/Stanford_Online_Products.zip" --output-dir "D:/data/processed_sop"
  python 02_SOP_Traditional_Baselines.py --data-dir "D:/data/processed_sop" --output-dir "D:/results/baseline_run1"
Optional protocol flags:
  python 02_SOP_Traditional_Baselines.py --all-train
  python 02_SOP_Traditional_Baselines.py --skip-official-test
Changing flags changes the protocol, so use a NEW output directory.

The original .ipynb files are retained as historical copies. They are no longer
required for the workflow. Both numbered .py scripts and both helper modules
contain English comments, descriptions and messages. Script 02 prints tables
to the terminal and saves plots as PNG files; it has no interactive-display
dependency. Importing the numbered scripts does not launch the workflow.
Existing completed preprocessing outputs remain compatible. Earlier baseline
checkpoints have a different helper-code fingerprint and must not be resumed
with the updated helpers; the new default result directory keeps them separate.

What script 02 implements
------------------
LR and RBF SVM; HSV48 (global colour) versus HSV + spatial gradient192.
The gradient feature is a simplified cell histogram, not full standard HOG.
This is a TRADITIONAL feature pilot, not the ResNet512 / region2048 experiment.
SupCon is not implemented here; all three algorithms remain required later.

Official train/test SKU partition is retained. At least 10,000 eligible original
training images are selected, retaining all images of each chosen product.
Products are selected by a seeded round-robin across the 12 categories.
Set use_all_official_train=True to use the full cleaned official training set.
The selected dataset size is not the number of sampled training pairs.
Default training cap: 1,000 anchor products -> at most 3,000 pairs per fold.
Default evaluation cap: 500 anchor products -> at most 1,500 pairs per fold.
This controls RBF SVM cost and must be reported as pair sampling, not full-data
training. Each image pair is contained within one product partition.

Manual category-stratified 10-fold OUTER CV on selected official TRAIN SKUs.
Manual 3-fold INNER CV on outer-training SKUs for C selection and thresholds.
No sklearn model_selection, metrics, GridSearchCV or automatic CV is used.
Only sklearn's classifier training is used. Scaling is manually fitted on
each relevant training fold. Each model has three C candidates; inner-only
range extensions are attempted when the winner is at a boundary. Read
grid_requirement_audit.json; unresolved ranges must be revisited honestly.
Do not force middle values to win or use outer-test scores to choose ranges.

Pair sampling: one same-SKU positive; one same-category random negative;
one same-category colour-hard negative per anchor. The latter is selected
from up to 50 other-product candidate images by CLEAN HSV48 distance, using
the same selection rule for every method. It is a colour-similarity proxy;
it does not prove products are perceptually similar. Positive pairs are
shared between the random and colour-hard reporting subsets. The combined
set has 1:2 positive/negative ratio; each separate subset has 1:1. Precision,
accuracy and AP therefore describe this sampled benchmark, not deployment
prevalence. The random negative excludes the same SKU but is not required
to be visually dissimilar. Pair manifests make the selection reproducible.

Train on clean images only. Perturb QUERY only; reference stays clean.
Five separate conditions, not combined corruptions:
  clean; occlusion25 (grey rectangle, approximately 25% content area);
  noise005 (Gaussian pixel noise, std=0.05 on the [0,1] scale, clipped);
  blur15 (Pillow GaussianBlur radius=1.5 pixels);
  rotation15 (fixed +15 degrees, fixed canvas, white fill; may crop content).
Corruptions affect the actual image content, leaving script 01's padding
unchanged. Seeds are derived from image IDs. All classifiers and feature
sets use exactly the same corrupted images, pairs and product folds.
No augmented image is counted toward the minimum 10,000 original images.
The one-level-per-corruption design is a first robustness experiment,
not a full severity curve. Add prespecified levels in a NEW run if needed.

C is tuned by mean clean-validation ROC-AUC across the three inner folds.
The operating threshold is chosen from selected-C pooled inner out-of-fold
clean scores to target 95% validation TPR, then fixed for every outer-test
condition. Test TPR may differ from 95%, and pooled inner scores need not
have exactly the final-refit score distribution. Report actual test TPR
and FPR together, rather than calling the test result 'FPR at 95% TPR'.
Each feature/model configuration has its own validation-derived threshold;
they are comparable at the same validation target, not guaranteed equal
test recall. Use paired FPR/TPR deltas to judge the matching tradeoff.

Metrics computed manually: TP/FP/TN/FN, accuracy, precision, TPR/recall,
FPR, F1, balanced accuracy, ROC-AUC (tie-aware ranks), average precision
(score-tie groups). Summary gives mean +/- 1 SD over 10 outer folds;
these are descriptive fold variability, not independent-sample confidence
intervals. Outer training sets overlap and pairs can share images/SKUs.
Figures show FPR, TPR, F1 and ROC-AUC with error bars. Pairwise ablation
deltas use the same outer fold, pair set, corruption and algorithm.

Optional final official holdout is enabled by default. It refits each
configuration on selected official-train products with NEW inner tuning,
then evaluates a fixed sample from official-test products. It is one extra
holdout, not the source of CV error bars. Do not use it for model selection.
SupCon and ResNet experiments should reuse protocol.json product folds,
saved pair IDs and the SAME corruption function for comparable results.

Teacher feedback and report reminders
------------------------------------
The teacher's feedback recommends robustness beyond partial occlusion:
noise, blur and rotations. All three are included in this implementation.
Describe why they model real image variation and discuss failures, not
only which algorithm has the largest accuracy.
The feedback also requires explaining how the chosen recent paper was
not discussed in class. For SupCon, distinguish the supervised contrastive
objective and its use of same-label positives from taught NN/CNN/ResNet
architectures and pretraining. Cite the selected paper, check the complete
course materials, and do not claim an approved paper title solely from
the screenshot: the screenshot confirms conference/year but names no paper.
Do not describe a frozen-feature adaptation as full end-to-end replication.

This code addresses the baseline evaluation requirements, NOT the whole
assignment: SupCon, at least two relevant literature papers, the four-page
report and team meetings still need to be completed.
No experiments or tests were executed when these files were authored.
Actual runtime, convergence, chosen C values and performance are unknown
until you run the scripts. Preserve logs and disclose actual data/pair
counts and limitations in the report.

Outputs
-------
protocol.json: parameters, products, folds, data/source fingerprints, versions.
experiment_manifest.csv: image-to-feature mapping and corrupted query coverage.
outer_fold_metrics.csv: every outer-fold metric, condition and negative subset.
summary_mean_sd.csv: means and SDs for the report.
paired_feature_deltas.csv / paired_delta_mean_sd.csv: effect of adding gradient.
all_tuning_records.json / grid_requirement_audit.json: inner tuning evidence.
figures/: four metrics with error bars for random and colour-hard negatives.
outer_XX/: pair manifests, inner tuning, saved models, per-pair predictions.
official_holdout/: supplementary official test outputs and final models.
feature_cache/: locally cached corruption features; do not submit these.

Pretrained ResNet explanation
----------------------------
ResNet need not be pretrained on SOP. Use public ImageNet-pretrained ResNet18
weights as a fixed feature extractor. Pretraining data relevance matters;
random/unrelated data is not equivalent to useful image pretraining.
If any weights are fine-tuned on SOP, fit them only on each training fold.
Never pretrain/fine-tune using all SOP products before claiming unseen-product
cross-validation. ResNet feature extraction will be a separate later step.
