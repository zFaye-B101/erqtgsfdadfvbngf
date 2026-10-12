# F5：白边对照（R1 补充）十折结果

2026-10-12，来源：用户从 Windows 电脑传回的 `results_F5.zip`。本开发电脑只核对文件和指标，没有重新训练。按[编号与版本规则](../../versioning.md)，F5 与 R1 使用同一运行名 `Main_Experiments`、同一代码指纹和同一协议，记为 **R1 的补充**，不另立轮次。

## 完成情况

F5（`F5_Traditional_Content_HSV_Gradient`）含两个特征：ContentHSV48、ContentHSV_HOG192。它们与 F1 HSV48、F2 HSV_HOG192 的计算完全相同，只是先按原图尺寸恢复商品内容矩形、再只在内容区域内统计，白色填边不计入。LR、RBF SVM、SupCon 均完成 10 个外层折、3 个内层折；协议、校准和五种扰动与 [R1](../2026-10-12-main-experiments/README.md) 相同，官方 holdout 未开启。

## 核对

- 300 份测试预测重算的混淆矩阵、TPR/FPR/F1、AUC/AP 与 900 行逐折指标一致；720 行均值与样本标准差一致；10 折完成记录一致；拟合、校准、测试商品互斥。
- 每个外层折的测试图片对（图片 ID、商品 ID、标签、负例类型）与 R1 中 F1 的完全相同。
- 代码指纹 `f5708eb4…` 与 R1 相同（对应仓库提交 `c275cd4` 中保留 Windows 换行的 `PythonCode/utils`）；协议 ID `68f23a72…` 与 R1 相同。
- 涉及 F5 的 14400 行跨组逐折差值从源指标重算一致，没有不兼容组被跳过。
- 回传包中的 F1–F4 文件共 2204 个，与 R1 归档的原始回传逐字节相同。
- 详见 [audit.json](audit.json)，逐文件哈希见 [file_manifest.json](file_manifest.json)；核对脚本为 [tools/audit_f5_archive.py](../../../tools/audit_f5_archive.py)。

## 主要结果

下表为 F5 减去对应带白边特征的逐折配对差值均值，括号内为 10 个外层折中差值大于 0 的折数（只描述方向，不是显著性检验）。

| 比较 | 算法 | 困难负例 原图 AUC | 随机负例 原图 AUC | 随机负例 遮挡 AUC | 困难负例 旋转 AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| ContentHSV48 − HSV48 | LR | −0.025（1/10） | +0.012（10/10） | +0.012（9/10） | −0.023（2/10） |
| ContentHSV48 − HSV48 | SVM | −0.015（1/10） | −0.005（4/10） | +0.005（7/10） | −0.003（6/10） |
| ContentHSV48 − HSV48 | SupCon | −0.011（1/10） | +0.002（5/10） | +0.005（7/10） | −0.009（2/10） |
| ContentHSV_HOG192 − HSV_HOG192 | LR | −0.005（5/10） | +0.010（8/10） | +0.017（8/10） | −0.038（0/10） |
| ContentHSV_HOG192 − HSV_HOG192 | SVM | −0.001（6/10） | +0.001（6/10） | +0.007（6/10） | −0.029（0/10） |
| ContentHSV_HOG192 − HSV_HOG192 | SupCon | −0.012（2/10） | +0.003（6/10） | +0.003（5/10） | −0.019（0/10） |

### 能得出的结论

- **去掉白边没有稳定改善手工特征。** 随机负例上 LR 有小幅且方向一致的提升（ContentHSV48 原图 +0.012，10/10 折），SVM 和 SupCon 基本不变。第一轮传统特征明显落后于 ResNet，不能归因于白边。
- **困难负例子集对 F5 有偏。** v3 协议的困难负例按“干净图、仅内容区域的 HSV48”选择，正是 ContentHSV48 本身；这些负例在 F5 的特征空间里被刻意选成最难分，因此困难负例上 F5 偏低（例如 LR −0.025，1/10）。比较 F5 与 F1/F2 时应以随机负例和全部负例为主，困难负例结果需注明此偏差。凡使用颜色特征的组，在困难负例子集上都会受到类似影响。
- **旋转条件下只用内容区域更差。** ContentHSV_HOG192 的旋转 AUC 在 LR、SVM、SupCon 上分别下降 0.038、0.029、0.019，10 折方向均一致。可能原因是旋转在内容矩形内留下白色角落，仅内容区域的统计会把它们计入商品；这一解释未经验证。
- **SVM 的 FPR 下降伴随 TPR 下降。** 例如 ContentHSV_HOG192 困难负例原图 FPR −0.042（0/10 折升高），TPR 同时 −0.017；属于固定阈值下工作点移动，不是净改善。完整逐折差值见 `artifacts/comparisons/`。

## 限制

- F5 的 SupCon 使用 CPU 版 torch `2.14.1+cpu`，F1/F2 的 SupCon 使用 CUDA（cu130）；数值可能有细微差异，比较时需注明。
- OpenCV 为 5.0.0，与 R1 相同。
- 没有官方 holdout、独立随机种子重复或显著性推断；与 R1 一样，完整预处理清单和全部内层划分未随结果回传。
- ResNet（F3/F4）仍使用带白边输入，本结果只回答手工特征的白边影响。

## 上传内容

- [关键指标](key_metrics.csv)、[全部指标](all_metrics.csv)、`artifacts/` 中 F5 的指标、预测压缩 CSV、校准/调参/完成 JSON、图表，以及包含 F1–F5 的 `comparisons/`（R1 原有比较的超集）。
- 两份 `protocol.json` 的四项本机路径改为相对路径；其余数据未改。276 个模型/检查点文件留在本机，未上传。
- 原始 ZIP 保留在本机；可用核对脚本从原 ZIP 复核：`python tools/audit_f5_archive.py --archive /path/to/results_F5.zip --output /path/to/evidence --source-code /path/to/c275cd4/PythonCode`。
