# 当前方向与待定事项

更新时间：2026-10-11。依据小组会话中的选择整理，不代表 staff 已审批或实验已完成。

课程核对补充见 [课程要求与staff反馈互评](reviews/2026-10-11_9291720_course-requirements_codex.md)：老师针对早期TabM提案建议在遮挡外增加其他鲁棒性设置；不等于批准后来SupCon方案。该记录仅供讨论，不在此擅自改变算法或正式协议。正式评价指标同样须自行实现，开发诊断的库函数结果不能原样替代。

## 已选择的方向

For unseen products, can adding local visual features reduce false matches between visually similar products under partial occlusion while maintaining same-product matching performance?

- 数据：Stanford Online Products (SOP)。`class_id` 是商品标注 ID；`super_class_id` 是大类。
- 比较：正则化 Logistic Regression、RBF SVM、SupCon-based matching。
- SupCon 训练使用商品 ID，使同商品表示靠近；最终用相似度和阈值判断两图是否同商品。不是用训练商品分类头直接预测未见商品。
- 早期 LightGBM、TabM、UCI HAR、SCARF 文件仅保留研究历史。

## 建议方案，尚待小规模验证

- 用户已要求按[SML课内测试安排](reviews/2026-10-11_ae5a1d6_feature-options_codex.md)分别检查池化、中间层和现有表示的训练/成本；DINOv2暂缓，不进入测试线。ResNet50/AE为后续备选，PCA按维度成本需要考虑。负责人及最终特征仍待确认，SupCon论文选择不变。
- [已完成开发诊断](experiments/2026-10-11-feature-diagnostics.md)已归档：含CNN逐种子CSV、环境/冒烟与传统特征汇总，缺失原始明细明确标注。只是冻结特征/单项评分测试，无正式三模型或嵌套评估结果。
- 新增的 `01_SOP_Preprocessing.ipynb` 当前会生成 192 维 HSV+空间 HOG 逐图特征；`sop_features.py` 提供 SIFT 图片对特征函数。这是可复用的初步预处理代码，不取代下列待验证的特征比较，也不表示正式评估协议已经确定。
- 固定外部预训练图像编码器，构造 global-only 与 global+local 两种特征条件。
- 以 2x2 区域池化作为局部特征候选，先不增加检测器；区域不等于语义部件。
- LR/SVM 使用图片对特征差异；SupCon 在每张图的特征上训练嵌入网络，再做匹配。
- 冻结骨干的轻量 SupCon 属于应用改编，不是端到端复现。训练用投影头与最终用于匹配的可训练表示必须区分，避免丢弃全部学习成果。
- 三算法各自比较两种特征，共六组配置；所有方法使用相同测试图片对和遮挡规则。
- 初始试跑可比较无遮挡与一侧图片遮挡面积 25%。遮挡位置、种子、填充值、训练是否增强仍待固定。
- 同大类不等于外观相似；拟用固定外部全局表示选择困难负例，同时报告普通负例，说明选择机制带来的偏向。
- 指标拟为 FPR、TPR、balanced accuracy，均自行计算；阈值只能使用训练/验证部分确定。

## 不可省略的评估边界

- 先按商品 ID 划分，再在各划分内部创建图片对；同一商品不得跨训练、验证和测试。
- 外层至少 10 折/重复，内层至少 3 折/重复，手动实现；三算法采用相同外层划分。
- 每算法至少调一个超参数，每个至少三个候选值；边界值经常胜出则扩展范围，不操纵结果强迫中间值胜出。
- 数据上学习的归一化、表示、模型、阈值均不得提前用外层测试集拟合。
- SupCon 在各训练折重新训练；固定外部编码器可缓存，不能用全部 SOP 训练后再声称独立交叉验证。
- 报告跨外层折/重复的误差条，不将相关图片对当作独立重复实验。
- 至少三项实验结果；重点是观察、解释和局限，不预设 SupCon 或局部特征必胜。

## 下一步

先锁定共同开发样本/配对清单、像素处理和指标，再分开测试池化与中间层；同步验证一轮LR训练及成本。Windows环境仍需实际核验。文献综述按数据集引用加至少两篇额外研究论文组织，说明SupCon相对课堂方法的新意，不要求排除课堂论文；见[原要求澄清](reviews/2026-10-11_9291720_course-requirements_codex.md)。尚无正式模型结果或staff批准凭证。
