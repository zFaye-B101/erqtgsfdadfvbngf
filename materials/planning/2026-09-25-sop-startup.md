> Historical document. Model and submission-status statements may be outdated. See the repository README and docs/current-plan.md for the current status.

# SOP 商品图像匹配：小组启动提案

状态：**待三名组员确认；尚未填入或提交官方模板。** 本文件取代同日的 UCI HAR + SCARF 候选计划。

## 已核实的起点

- 数据：作者发布的 Stanford Online Products (SOP) 原包已下载；120,053 张图中 120,026 张原图满足宽、高均不小于 100 像素。27 张不合格图片直接排除。详见[本地核验记录](../research/stanford-online-products-audit.md)。
- 官方要求：研究问题不能只是预测现成标签，也不能只是比较三个算法；必须有实质性图像特征构造，三个不同复杂度的算法，共用自行实现的外层评估与内层调参。详见[Project.pdf](../official/Project.pdf)。
- 作者的 [SOP 论文](https://cvgl.stanford.edu/papers/song_cvpr16.pdf) 已研究一般的困难负例挖掘，并指出细微外观差异导致的检索失败。因此不把“困难负例是否有用”本身当作新问题。

## 推荐研究问题（Section 2.a）

**中文：** 在未见过的商品上，加入局部区域视觉特征，能否在部分遮挡时减少外观相似、但不是同一商品的误匹配，同时保留同一商品的匹配能力？

**English:** For previously unseen products, do region-aware visual features reduce false matches between similar-looking but distinct items under partial occlusion, without sacrificing same-item matching performance?

这不是对 22,634 个现成类别直接做分类。任务是给两张商品图判定是否为同一标注商品，并分析“仅全局特征”与“全局加局部特征”在干净、遮挡两种条件下的取舍。同一大类的不同商品作为重点分析的负例，不把“同大类”直接等同于“外观相似”。

## 暂定方法（Section 2.c-2.g）

1. 从合格原图选约 12,000-20,000 张，保留足够多且每类至少两张的商品。使用相同的固定图像表征方法构造全图和局部区域特征；对图片对再构造距离、相似度、差值等配对特征。只缩放图片不算特征构造。若使用外部预训练权重，记录来源、固定权重，并在报告中明确披露。
2. 三个不同模型共用相同的数据划分和可比输入：正则化 Logistic Regression（simple）、RBF SVM（medium）、LightGBM（complex）。LightGBM 对应 [Ke et al., NIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)；其 GOSS/EFB 是该论文提出的关键技术，正式实现应确认所用配置确实启用对应机制。SupCon（NeurIPS 2020）是更偏视觉学习、但计算开销更高的备选，暂不写进提交稿。
3. 按 `class_id` 整组划分，避免同一商品同时进入训练与测试。计划外层 10-fold、内层 3-fold，自行写划分和调参，不使用第三方 CV/搜索器。配对样本与任何需从 SOP 学得的变换只能在各自训练折内建立；三种模型使用同一批外层折。
4. 在外层测试折分别测原图和预先固定规则的部分遮挡图，所有模型使用同样的遮挡。报告同商品匹配率、误匹配率、同大类困难负例结果、运行成本及跨折误差条；评价指标自行计算。具体阈值在训练/验证部分确定，不能看外层测试标签来调。

## 今天的 Questions.docx 草稿范围

官方模板只要求 Section 2.a 的暂定研究问题和 Section 2.d 的暂定论文，不需要今天写完完整实验方案。建议在组内确认后使用上面的英文问题，并在论文栏填入：

> Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems 30 (NIPS 2017). We plan to compare LightGBM with regularized logistic regression and RBF SVM on constructed image-pair features. Could staff confirm that this paper and intended use satisfy the Section 2.d recent-algorithm criterion?

研究问题栏可补充向 staff 确认：上述“未见商品 + 局部特征 + 遮挡误匹配”的问题是否足够超出直接预测 SOP 类别。不要把这份计划写成已获 staff 批准。

## 今日顺序

1. 三人确认研究问题、使用 LightGBM 还是更昂贵的 SupCon，以及谁负责两个 DOCX。
2. 用官方 `Questions.docx` 填入三人姓名、研究问题和论文；`GroupAgreement.docx` 的协作条款需三人共同同意。
3. 一人分别上传两个 DOCX，另一人核对 Canvas 的实际提交状态。当前本地提案和文件准备都不等于已提交。
