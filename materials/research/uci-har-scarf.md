> Historical document. Model and submission-status statements may be outdated. See the repository README and docs/current-plan.md for the current status.

# Candidate: UCI HAR + SCARF

## Dataset

- UCI Human Activity Recognition Using Smartphones
- 10,299 instances、561 published features、30 subjects、6 activities
- 同时包含长度为 128 的原始惯性信号窗口，可用于自行构造特征
- 数据集 URL：`https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones`

## Recent paper

- Dara Bahri, Heinrich Jiang, Yi Tay, Donald Metzler
- *SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption*
- ICLR 2022
- 官方页面：`https://openreview.net/forum?id=CuV_qYkmKb3`

## Proposed design

- Outer evaluation：按 subject 分组的 10-fold CV，自行实现。
- Inner tuning：按 subject 分组的 3-fold CV，自行实现。
- Feature construction：每个传感器轴的统计量、能量、FFT band energy/spectral entropy、向量幅值和轴间相关性。
- Labelled-data conditions：例如 10%、30%、100%，抽样只发生在 outer-training fold 内。
- Algorithms：Logistic Regression、RBF SVM、SCARF encoder + MLP classifier。
- Metrics：accuracy、macro-F1、per-class recall/confusion matrix，全部自行计算；报告 outer folds 的均值和 error bars。

## Leakage controls

- 同一 subject 的样本不能同时进入 outer train 与 outer test。
- 标准化参数、特征选择和 SCARF 的 feature-marginal corruption distribution 都只能从当前 training fold 得到。
- 调参只能使用 inner folds，不能观察 outer test fold。

## Main risk

SCARF 的训练次数叠加 nested CV 后仍有计算量。先用少量 epoch 和一个 fold 做 pilot，再确定隐藏层宽度、corruption rate 与训练预算；不要一开始铺开全部 10 个 outer folds。
