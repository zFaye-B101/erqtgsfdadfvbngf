# 90% 校准召回：R1/R2 的工作点补充分析

2026-10-13 核对用户回传的 `results_recall90.zip`，覆盖 `Main_Experiments`（F1–F5）及 `ResNet_LocalFeature_Comparison`（F3/F4/F4.1/F4.2）。不是新特征组或训练轮次。原95%结果保留不动。

## 如何解读

仅使用每个已训练模型的干净独立校准集正例重新选阈值，目标从95%改为90%；特征、模型、超参数、测试配对与分数不变。AUC/AP 不因阈值改变；FPR 降低通常伴随 TPR 降低。

以新局部比较的遮挡条件、全部负例为例（百分比）：

| 模型/特征 | 95%目标 TPR | 95%目标 FPR | 90%目标 TPR | 90%目标 FPR |
| --- | ---: | ---: | ---: | ---: |
| LR / F3 | 92.66 | 70.08 | 85.73 | 51.82 |
| LR / F4.2 | 92.03 | 64.01 | 85.26 | 47.79 |
| SupCon / F3 | 90.62 | 57.50 | 82.92 | 40.76 |
| SupCon / F4.2 | 90.52 | 55.10 | 82.03 | 37.84 |

这是工作点取舍，不是额外的模型学习收益，也不保证测试召回达到90%。原95%目标与R2判断标准仍为原主分析；90%另列，不能选择更有利版本替换原结果。

## 核对内容

- 18个算法族结果目录，1500份测试预测、4500行逐折指标、3600行均值/样本标准差。全部八指标及混淆矩阵重算一致。
- 300份阈值按校准正例重算一致；每份校准/测试分数及配对与原95%结果核对一致（数值容差 `rel_tol=1e-9, abs_tol=1e-10`），`original_threshold_95` 与原值一致。
- 逐折 `comparison_95_vs_90.csv` 的原/新指标及六项差值重算一致。分割互斥使用原95%校准记录中的拟合/校准商品列表核对。
- 主实验原源码对应 `c275cd4`，新局部比较对应 `f412219` 的模型源码。重校准脚本 SHA256 为 `380b98db2a5d73958bee044b38c81c41152bf111478dc2b182b90abbd3aa41f1`，与仓库 `03_SOP_Recalibrate_90.py` 字节匹配。
- 本次还重算了原95%包的1500份预测用于对照；旧R1/F5归档不改写。未加载模型、未重新训练，也未使用官方holdout。

## 上传范围

- [all_metrics.csv](all_metrics.csv)、[outer_fold_metrics.csv](outer_fold_metrics.csv)、[key_metrics.csv](key_metrics.csv) 同时含两个运行，必须按 `run` 区分。
- `artifacts/` 共2272份，保留两个运行的逐折与汇总CSV、95/90对照、压缩预测、校准/协议JSON和诊断图，以及根级 `recalibration_info.json`。
- [local_feature_decisions.csv](local_feature_decisions.csv) 将原描述性规则应用于90%结果，供补充分析；不是事先设定的90%确认性检验。
- [audit.json](audit.json)、[file_manifest.json](file_manifest.json) 记录复核与文件哈希。JSON中的绝对路径递归脱敏，尤其包括 `original_protocol.config` 与重校准源目录。
- 排除原图、数据ZIP、缓存和权重。模型/调参来源在原95%归档，不把这份后处理包伪装成新的训练证据。

复核命令见[R2归档](../2026-10-13-R2-local-features/README.md)。本次只上传结果证据及维护报告提纲，最终四页报告和交付包仍需整合。
