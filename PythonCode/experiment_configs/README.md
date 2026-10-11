# Saved feature groups

Run `02_SOP_Experiments.py`; every JSON here is discovered automatically.
Common settings and paths live in `../config.json`. Each group
specifies a unique `group`, optional `description`, and `features` mapping.
Complete merged settings are recorded in each result family's protocol.json.

Current groups: F1 HSV48; F2 HSV plus spatial gradient192; F3 ResNet global512;
F4 ResNet global plus four regions2560. All use padded images.
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

Registry: `hsv_hog`, `hsv_hog_content`, `resnet18`. A new fixed method needs
implementation and registration in `utils/sop_feature_bank.py` plus a JSON.
Data-fitted methods such as PCA must fit inside training folds, not global caches.

Groups inherit common folds, pairs and corruption settings. Change common rules
deliberately and choose a new protocol_dir shared by all compared groups.
Keep old configurations, source versions and environments for reproduction.
Cross-group reports check protocol, code and common configuration compatibility.
Retired paired configurations and launchers have been removed.
