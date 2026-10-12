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

Naming (full rules in docs/versioning.md of the repository): a new method gets a new number (F7, F8, ...); an improvement of an
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
