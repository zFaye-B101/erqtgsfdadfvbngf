# Saved feature groups

Run `02_SOP_Experiments.py`; every JSON here is discovered automatically.
Common settings and paths live in `../config.json`. Each group
specifies a unique `group`, optional `description`, and `features` mapping.
Complete merged settings are recorded in each result family's protocol.json.

Current groups: F1 HSV48; F2 HSV plus spatial gradient192; F3 ResNet global512;
F4 ResNet global plus four regions2560; F5 content-only HSV48 and HSV plus
spatial gradient192. F1-F4 use padded images. F5 uses hsv_hog_content, which
excludes the white padding; compare F5 with F1/F2 to measure the padding effect.
Its gradient cells resize the content rectangle to 128x128 without keeping the
aspect ratio. F6 is frozen ResNet50 (IMAGENET1K_V2) global average pooling,
2048 dimensions, on the same padded 224 input as F3; compare F6 with F3 to measure
the backbone change. Judge local features only within one backbone: a later
ResNet50 global-versus-local group can use resnet50_regions, whose columns
[0, 2048] equal the F6 descriptor. Do not compare ResNet50 global with ResNet18
local to decide whether local features help.
The gradient descriptor is a simplified cell histogram, not full standard HOG.

## Local aggregation groups A and B

Plan A has one canonical configuration: F4.1. The duplicate F7 is removed.
Former F8 is renamed F4.2 for plan B. Existing result folders are preserved.

F6 remains ResNet50 global2048; it is not renamed or overwritten.
F4.1_ResNet_GlobalRegionMax1024 (A) retains F4's global512 and takes the channelwise
maximum of its four regional mean descriptors, producing another512 dimensions.
Its component region_max transform uses the verified resnet18 source cache.
Old-version caches are not silently adopted when source fingerprints differ.
F4.2_ResNet_GlobalCropMax1024 (B) retains the same global512 and separately encodes five
160x160 windows (four corners and centre) resized to224 with PIL bilinear sampling.
The five global-average descriptors are channelwise-max aggregated into local512.
Both groups retain ImageNet V1 weights, padded inputs and existing fold scaling;
no descriptor L2 or new fusion weights are added. B uses six sequential image
batches per batch of originals, including the global batch, to limit peak memory.
Query corruption is applied before cropping, without using corruption masks.

Compare F3,F4,F4.1,F4.2 in round R2 (docs/versioning.md):
python 02_SOP_Experiments.py --group F3,F4,F4.1,F4.2 --method all --run-name R2_LocalFeatureAB
Report all four corruptions and clean results, TPR/FPR/AUC and fold differences.
F3/F4 are 512/2560 dimensions; F4.1/F4.2 are1024. A/B isolate different local encoding
approaches better than comparison with F3 alone, but differ in crop geometry
and context. No accuracy or FPR improvement has been verified on real SOP data.
Adding these extractors changes cache/model fingerprints: do not resume old
group folders under the new code. Script01 data can still be reused.
New F4.1/F4.2 folders can use default Main_Experiments with --group F4.1,F4.2 --method all.
Automatic comparisons with old-version groups are skipped; this does not prevent
training or saving the new groups. Use one new run/version for automatic paired
comparisons including F3/F4.

## Add an existing-feature combination

Create `F5_HSV_ResNetGlobal.json`, for example:

```json
{
  "group": "F5_HSV_ResNetGlobal",
  "description": "HSV and frozen ResNet global with per-block L2 normalization",
  "features": {
    "HSV_ResNet560": {
      "components": [
        {"extractor": "hsv_hog", "columns": [0, 48], "normalization": "l2"},
        {"extractor": "resnet18", "columns": [0, 512], "normalization": "l2"}
      ]
    }
  }
}
```

This example is not an active F5 file. Normalization is `none` (default) or
`l2` per image per component. Components concatenate in the specified order.
Simple extractor/columns specifications still work. Column end is exclusive.
Existing fold-fitted model scaling follows feature extraction. Labels are not features.

F4.1 (plan A) keeps the F3 global 512 columns and reduces the four F4 region
vectors (columns 512-2560) by a per-channel maximum across regions, 1024 total.
It reuses the same ResNet18 extraction as F3/F4 in one run, so no extra CNN pass.

Naming (full rules in docs/versioning.md of the repository): a new method gets a new number (F4.1, F4.2, ...); an improvement of an
existing method gets a sub-version of that method (F4.1, F4.2, ...). Never edit
an existing group JSON whose results are archived; add a new file instead.
Groups are listed and compared in version order (F4, F4.1, F4.2, F5), and
paired deltas read later minus earlier (for example F4.1 minus F4).

Component transform: `"transform": "region_max"` with `"regions": n` splits the
selected columns into n equal region vectors and keeps the per-channel maximum
(applied before optional l2 normalization). Default transform is `none`.

Registry: `hsv_hog`, `hsv_hog_content`, `resnet18`, `resnet50` (2048) and
`resnet50_regions` (2048 global + 8192 regions = 10240). A new fixed method needs
implementation and registration in `utils/sop_feature_bank.py` plus a JSON.
Data-fitted methods such as PCA must fit inside training folds, not global caches.

Groups inherit common folds, pairs and corruption settings. Change common rules
deliberately and choose a new protocol_dir shared by all compared groups.
Keep old configurations, source versions and environments for reproduction.
Cross-group reports check protocol, code and common configuration compatibility.
Retired paired configurations and launchers have been removed.
