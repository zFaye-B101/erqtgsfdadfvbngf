# R2：局部特征正式比较（95% 校准目标）

2026-10-13 从用户提供的 `results.zip` 核对并归档。实际运行名为 `ResNet_LocalFeature_Comparison`，在登记表中对应 R2；保留原目录、协议和指纹，不改成计划运行名 `R2_LocalFeatureAB`。本机只重算和核对已有输出，没有训练或加载模型。

## 结果与判断

F3、F4、F4.1、F4.2 × LR/SVM/SupCon，均为 10 个外层商品折、3 个内层折、独立校准集、95% 目标召回。五条件为原图、遮挡25%、噪声0.05、模糊1.5、旋转15度。官方 holdout 未使用。

**独立窗口编码的收益具有算法依赖性。** F4.2 相对 F3 的 LR 在全部负例及随机负例的五个条件下，均满足此前记录的描述性规则：FPR 均值下降且至少 8/10 折改善，TPR 差值均值不低于 −1 个百分点。不是显著性检验。

下表为全部负例，F4.2 减 F3；TPR/FPR 单位为百分点，AUC 为原始数值差。

| LR 条件 | ΔFPR | ΔTPR | ΔAUC | FPR 改善折数 |
| --- | ---: | ---: | ---: | ---: |
| 原图 | −7.53 | −0.68 | +0.0187 | 9/10 |
| 遮挡 | −6.07 | −0.63 | +0.0181 | 8/10 |
| 噪声 | −6.02 | −0.57 | +0.0134 | 8/10 |
| 模糊 | −7.11 | −0.99 | +0.0221 | 8/10 |
| 旋转 | −4.87 | −0.47 | +0.0234 | 8/10 |

- 模糊条件的 TPR 接近允许下降界线，不能写成“完全没有召回损失”。
- F4.2 的 SupCon 在全部负例的噪声条件满足规则；随机负例另有遮挡满足。SVM 未达到至少8折同向的联合规则。三算法五条件的平均 AUC 均提高，不意味着每折提高或统计显著。
- F4.1 的 LR 在全部负例上原图、遮挡、噪声满足；随机负例上原图、遮挡满足。F4 的固定区域拼接在这两个子集均无满足项。
- 全部算法、条件和负例子集均保存在 [local_feature_decisions.csv](local_feature_decisions.csv)，不只列有利结果。

## 核对与来源

- 600 份测试预测重算全部八指标及混淆矩阵，与 1800 行逐折指标一致；1440 行均值和样本标准差一致；80 份算法族完成记录一致。
- 10800 行跨组配对差值重算一致；同折各组、算法和条件测试图片对一致。
- 120 份模型的拟合/校准/测试商品集合互斥；校准阈值从保存的正例分数独立重算。完整预处理清单、全部内层成员不在回传包内。
- 代码指纹 `84942fc741db045250ffc3c6a4cfd597e35c52df64c0714042d5a66cc969dbf3`，与 `f412219` 的 `PythonCode/utils` 字节匹配。共同协议 `68f23a72ae919ce97a9933ab8c5b5594ae824314262bbc9a6c6d322763894e45`。
- 此轮保存的 ResNet 和 SupCon 环境记录为 CPU torch `2.14.1+cpu`。不要沿用旧 R1 的 CUDA 描述；主要比较使用本轮一起重跑的 F3/F4。
- R1 外层结果已用于提出 A/B，R2 属于探索性比较。F4.2 同时改变局部视野、重采样和表示维数，不能把收益归因于单一机制。

## 文件与复核

- [all_metrics.csv](all_metrics.csv)：均值与样本标准差；[outer_fold_metrics.csv](outer_fold_metrics.csv)：逐折明细；[key_metrics.csv](key_metrics.csv)：原图/遮挡摘要。
- `artifacts/ResNet_LocalFeature_Comparison/`：原始指标、压缩预测、校准/调参/完成 JSON、组级诊断图和配对差值，共1124份。已有的训练loss轨迹未发布，不影响预测复核。
- [file_manifest.json](file_manifest.json) 保存源文件与公开副本 SHA256；[audit.json](audit.json) 保存检查范围与排除清单。JSON 中机器绝对路径已递归转为便携路径（含嵌套配置）；其他字节保留。
- 数据集、缓存、原 ZIP、模型权重、训练检查点均未上传。公开证据不是可续训目录，也不是最终提交代码包。
- 重建工具：[archive_local_results.py](../../../tools/archive_local_results.py)，仅 Python 标准库；同时验证两份包并要求输出目录不存在：

```sh
python tools/archive_local_results.py --source /path/to/results.zip --recall90 /path/to/results_recall90.zip --output-local /tmp/local-evidence-new --output-recall90 /tmp/recall90-evidence-new
```

需要仓库历史 `c275cd4`（R1 源码）及匹配 R2 指纹的 `PythonCode/utils`。代码后续变化时，应在本次归档提交的 checkout 中运行，不绕过指纹检查。90% 版本另见[补充归档](../2026-10-13-R1-R2-recall90/README.md)。
