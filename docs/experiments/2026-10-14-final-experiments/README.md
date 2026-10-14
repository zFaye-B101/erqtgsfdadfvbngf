# 最终八组实验结果（2026-10-14）

## 范围与使用

最终特征组为F1、F1.1、F2、F2.1、F3、F4、F4.1、F4.2。Main_Experiments包含前六组；ResNet_LocalFeature_Comparison包含F3/F4/F4.1/F4.2。F3/F4在两个批次出现，不是额外方法。共30个特征—算法组合、300份外层最终模型记录；每目标1500份测试预测、4500行逐折指标。

- [95%汇总](recall95/all_metrics.csv)、[逐折指标](recall95/outer_fold_metrics.csv)、[局部增量判断](recall95/local_feature_decisions.csv)。
- [90%汇总](recall90/all_metrics.csv)、[逐折指标](recall90/outer_fold_metrics.csv)、[局部增量判断](recall90/local_feature_decisions.csv)。
- [白边配对差值](padding_comparisons.csv)：F1.1−F1、F2.1−F2，覆盖两个目标、三算法、五条件、三负例口径、八指标，共1440行。TPR/FPR差值乘100后为百分点；负值表示下降，是否有益须联合判断。
- [95%核对记录](recall95/audit.json)、[90%核对记录](recall90/audit.json)，源文件与发布文件SHA256见各目录file_manifest.json。artifacts保留原运行目录层次及指标、压缩预测、协议/调参/校准记录和诊断图。

来自本地Code/results.zip和Code/results_recall90.zip。原ZIP、数据、权重、缓存和训练loss轨迹不上传。JSON机器绝对路径去除；清单记录原始与发布字节哈希。95%是主分析，90%只改变工作点，不是新训练或新F组。新快照无F5/F1.2；历史归档保留原名。

## 独立核对与版本边界

使用标准库工具tools/archive_final_results.py及archive_local_results.py重算所有八项指标、混淆矩阵、校准正例阈值、汇总均值与样本标准差，并检查拟合/校准/测试商品隔离、协议与预测对一致性。90%和95%分数及图片对逐项相同。03源码哈希允许Git检出造成的LF/CRLF差异，不允许其他源码差异。Windows输出使用长路径前缀。

Main原F1/F2/F3/F4使用旧指纹f5708eb496addc1ea87413327cf464b5bd0b8428b7d87c437313bf3cafccd5b4；新F1.1/F2.1和局部比较使用84942fc741db045250ffc3c6a4cfd597e35c52df64c0714042d5a66cc969dbf3。共同协议ID为68f23a72ae919ce97a9933ab8c5b5594ae824314262bbc9a6c6d322763894e45。

自动比较保留跨指纹跳过保护。白边补充表是在测试对相同、相关特征/算法/协议实现差异经审阅后计算的跨版本描述性对照，不冒充同版本自动比较。未加载模型、重训或全面复审预处理清单及内层商品归属；官方holdout未使用。折间标准差不是置信区间。

## 主要解读与报告组织

干净混合负例上F4.2三算法AUC均值最高（约0.844–0.846）；不同算法接近，不主张SupCon显著优于SVM。四类扰动均导致该表示AUC下降；FPR下降须同时查看TPR，不能当作单独鲁棒性证据。

95%目标下F4.2−F3的LR在混合/随机负例全部五条件满足原描述性标准：FPR均值下降且至少8/10折下降，TPR均值下降不超过1个百分点。SupCon仅部分条件支持，SVM未达同一标准。该标准不是统计显著性或等效检验。

白边去除未带来跨方法、跨条件的一致收益。90%阈值目标降低误匹配，同时降低测试召回；AUC和AP不变。报告4.1写总体及扰动，4.2写局部构建和独立白边对照段，4.3写阈值，4.4写局限。历史F5结论以此快照的正式F1.1/F2.1名称表述。

## 复现归档

在仓库根目录运行：

```text
python tools/archive_final_results.py --source /path/to/results.zip --recall90 /path/to/results_recall90.zip --output /path/to/new-output
```

输出必须不存在；输入只读。此工具是结果核对/归档工具，不是GPU实验入口。此次未修改PythonCode或个人报告文件。
