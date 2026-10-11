# 当前方向与待定事项

更新时间：2026-10-11。依据小组会话中的选择整理，不代表 staff 已审批或实验已完成。

## 已选择的方向

For unseen products, can adding local visual features reduce false matches between visually similar products under partial occlusion while maintaining same-product matching performance?

- 数据：Stanford Online Products (SOP)。`class_id` 是商品标注 ID；`super_class_id` 是大类。
- 比较：正则化 Logistic Regression、RBF SVM、SupCon-based matching。
- SupCon 训练使用商品 ID，使同商品表示靠近；最终用相似度和阈值判断两图是否同商品。不是用训练商品分类头直接预测未见商品。
- 早期 LightGBM、TabM、UCI HAR、SCARF 文件仅保留研究历史。

## 建议方案，尚待小规模验证

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

确认 Windows 环境与数据路径，跑通小数据子集；测量一轮完整训练和验证成本后，再锁定样本量、采样规则、嵌入结构、超参数及正式协议。尚无正式模型结果或 staff 批准凭证。
