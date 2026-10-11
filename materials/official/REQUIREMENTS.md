# A2 官方要求核对

核对来源：2026 S2 Canvas `Project.pdf`（4 页）、两个提交页面与官方 DOCX 模板。`GroupAgreement.docx` 和 `Questions.docx` 的未修改原件已保存于本目录。`guidance/墨大COMP90051A2P1.pptx` 是第三方辅导资料，备注说明其底稿来自 2024 S2，因此只能作为次级参考。

## 2026-09-25 23:59 前

这是官方提供的 optional、not graded checkpoint；本组若按计划使用它，需要分别提交两个 DOCX。两项均允许 2 次尝试，任一成员提交后计入全组：

1. `GroupAgreement.docx`
   - 三名成员姓名
   - Goals：项目目标与希望提升的技能
   - Expectations：会议出席、参与、沟通频率、工作质量
   - Policies and Procedures：协作规则
   - Consequences：未履行约定时如何处理，包括期末 peer self-assessment 记录
2. `Questions.docx`
   - 三名成员姓名
   - Section 2.a：计划采用的 research question，可附澄清问题
   - Section 2.d：计划采用的 conference/journal paper，可附澄清问题

## 方法要求（Section 2）

### 2.a Research question

- 不能只是对数据集标签做显而易见的预测。
- 不能只是比较 3 个算法。
- 应研究表示、鲁棒性、标签稀缺、偏差、跨域泛化、零样本等更具体的问题。

### 2.b Dataset

- 必须已公开、可及时访问；报告中给出 URL。
- Tabular：至少 10,000 instances、100 features；原始特征与合理构造特征均可计入。
- Image：至少 10,000 张，原图至少 100 x 100；不能靠复制或放大凑规模。
- 可以取大数据集的较小子集，以控制交叉验证和调参成本。

### 2.c Feature construction and preprocessing

- 仅做 one-hot、标准化/插补、去重、采样、表连接、缩放图片或普通文本处理不够。
- 不能直接把单表或图片原样送入算法；需要有实质性的特征构造或预处理。

### 2.d Algorithms

- 比较 3 个不同算法，复杂度应覆盖 simple、medium、complex。
- 不能用同一算法的不同超参数冒充 3 个算法。
- 至少一个算法未在课上讲过，并首次发表于 2017 年或以后。
- 合格 venue：NeurIPS、ICML、UAI、AISTATS、JMLR、ICLR、TMLR；这些会议的 workshop 不算。

### 2.e Cross-validation

- 必须自行实现，不能调用第三方 cross-validation 工具。
- 不能只用一个 test set；3 个算法必须采用相同的外层评估方式。
- Repeated train/test splits：至少 10 次。
- k-fold：`k >= 10`。
- Bootstrap：`B >= 10`。
- 时间序列必须按过去预测未来，不能随机打乱造成泄漏。

### 2.f Hyperparameter tuning

- 每个算法至少调 1 个超参数，建议每个算法不超过 3 个。
- 必须自行实现 tuning 与 nested cross-validation，不能调用第三方调参或 nested-CV 工具。
- 每个超参数至少尝试 3 个值；最终常被选中的值应位于搜索范围中部，否则要扩展范围。
- 内层 train/validation splits 至少 3 次；inner k-fold `k >= 3`；bootstrap `B >= 3`。

### 2.g Results

- 指标必须自行计算；可以用第三方库画图。
- 至少展示 3 项实验结果，并报告 error bars。
- epoch/iteration 对 loss 的曲线不算 learning curve；learning curve 应改变训练样本数量。

## 最终提交（2026-10-16 23:59）

- 单栏 A4、最多 4 页的 PDF research report；10 pt 字号，四边 1.5-2 cm。
- 参考文献和附录也包含在 4 页内，不要封面；超过第 4 页的内容不评分。
- ZIP：Python code 或 Jupyter notebook、简短 `Readme.txt`、自动化脚本。
- 不提交数据。
- 至少 3 次小组会议；可附 meeting minutes 与 Slack/GitHub 日志。

报告内容：成员信息、Introduction、Literature review（数据集文献及至少 2 篇额外论文）、Methods、Results and Discussion、Conclusion、Bibliography。

评分：Completeness and correctness 12 分、Clarity and structure 6 分、Critical analysis 12 分。仅说“模型 B 准确率更高所以更好”会被扣分；需要解释原因、权衡、局限、替代方案及未采用理由。

## 与课程内容的直接联系

- Week 1-4：风险、泛化、正则化、SVM，支持基线模型选择与结果解释。
- Week 5-7：kernel、perceptron、deep learning、CNN/RNN/Transformer，支持 medium/complex 模型与表示学习。
- Week 8：Cross-Validation 是本作业最关键的实现依据，尤其是 nested CV、统一评估协议和避免数据泄漏。
- Week 9 内容并非启动材料的硬性前置；除非选 bandit 或 Bayesian regression 方向，否则不用为今天的选题等待它们。

## 2026-09-27 Canvas 复核

- 已与 Canvas 作业页 `Group project report and code` 复核：截止 **2026-10-16 23:59**，30 分，`online_upload`。上述要求没有变化；作业页原文保存在 `canvas-page-2026-09-27.md`。
- Canvas 上的 `Project.pdf`（file 28749213，160,464 bytes，更新于 2026-09-25）与本目录 `Project.pdf` 大小一致。
- 新增官方会议记录模板 `MeetingMinutes.docx`（file 28749212），可选附在提交中，用于证明至少开过 3 次会。
