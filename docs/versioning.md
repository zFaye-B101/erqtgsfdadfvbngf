# 编号与版本规则

## 90%阈值补充分析（2026-10-12）

新增 `PythonCode/03_SOP_Recalibrate_90.py`，复用保存的校准和外层测试分数，
不重新训练、不改95%公共配置。阈值仅用每折干净独立校准集正例确定。
运行 `python 03_SOP_Recalibrate_90.py`，默认读同目录 results.zip，否则读 results/；
输出独立 `results_recall90/`，已有目录一律拒绝覆盖。可用 --source 与 --output 指定位置。
输出保留原运行名/组别/模型类别，含八项指标、图表、预测和95%对90%对照。
这不是新F组或训练轮次，原数据/模型/95%结果不变。90%目标不保证测试召回90%或FPR<50%。


用户于 2026-10-12 确定的统一规则，适用于全部协作者与 AI。只约束今后新增的编号；已归档的结果、目录和配置不改名，只在下方登记表中对应。

## 1. 特征组（方法）编号：F

- **新方法**用新的整数编号：F7、F8……；编号只增不复用。
- **已有方法的改进**用该方法的子版本：F4.1、F4.2……（例如局部特征方案 A 是对 F4 的改进，记为 F4.1）。
- 组名格式为 `F<编号>_<英文描述>`，例如 `F4.1_ResNet_GlobalRegionMax1024`；配置文件名与组名相同，放在 `PythonCode/experiment_configs/`。
- 已有结果对应的组文件不修改；要改就新建子版本或新编号。
- 列表与跨组配对差值按版本顺序排列（F4 → F4.1 → F4.2 → F5），差值读作“后者减前者”。
- 讨论阶段可用“方案 A/B/C”；一旦实现，文档和结果一律以 F 编号为准，并在本表登记对应关系。

## 2. 实验轮次编号：R

- 每一轮正式运行（同一版代码、同一运行名下的一组实验）分配一个轮次编号 R1、R2……，只增不复用。smoke 和合成测试不占轮次编号。
- **运行名**（`--run-name`）格式为 `R<轮次>_<主题>`，主题用英文驼峰、不含点，例如 `R2_LocalFeatureAB`。代码会拒绝运行名中的点。
- 代码指纹变化后，需要配对比较的旧组必须在新轮次中一起重跑；不同轮次的结果不直接配对。
- **结果存档目录**格式为 `docs/experiments/<YYYY-MM-DD>-R<轮次>-<主题小写连字符>/`，例如 `2026-10-13-R2-local-feature-a/`，日期取结果回传日。

## 3. 其他编号

- 统计后处理不分配新F/R号；`2026-10-13-course-statistics`是跨95/90工作点的课堂统计补充目录，不是新的正式实验轮次。

- 互评记录中的问题编号 R1、R2……只在该记录内部有效，与实验轮次无关；引用时写明记录文件，例如“局部特征互评 R2”。
- 提交说明（commit message）使用中文，可保留 `feat:`、`fix:`、`docs:` 等类型前缀，例如 `feat: 新增 F4.1 跨区域最大汇总`。2026-10-12 之前的英文提交不改写。

## 4. 登记表

| 编号 | 内容 | 运行名 / 位置 | 状态 |
| --- | --- | --- | --- |
| F1 | HSV48 | `F1_Traditional_HSV48` | R1 已完成 |
| F1.1 | 仅内容区域HSV48，原F5的颜色分支 | `F1.1_Traditional_Content_HSV48` | 配置已拆分；正式重跑待执行 |
| F2 | HSV＋空间梯度192 | `F2_Traditional_HSV_Gradient192` | R1 已完成 |
| F2.1 | 仅内容区域HSV＋梯度192，原F5的梯度分支 | `F2.1_Traditional_Content_HSV_Gradient192` | 配置已拆分；正式重跑待执行；02中的F1.2仅为输入兼容别名 |
| F3 | ResNet18 全局512 | `F3_ResNet_Global512` | R1 已完成；R2、R3 中重跑 |
| F4 | ResNet18 全局＋2×2区域2560 | `F4_ResNet_GlobalLocal2560` | R1 已完成；R2 中重跑 |
| F4.1 | 局部特征方案 A：全局512＋跨区域最大512 | `F4.1_ResNet_GlobalRegionMax1024` | R2 已完成并归档 |
| F4.2 | 局部特征方案 B：全局512＋五窗口独立编码最大512 | `F4.2_ResNet_GlobalCropMax1024` | R2 已完成并归档 |
| F5 | 历史合并白边组：HSV48/HSV＋梯度192 | `F5_Traditional_Content_HSV_Gradient` | 活跃配置已删除，由F1.1/F2.1替代；原R1存档保留 [2026-10-12-R1-f5-padding](experiments/2026-10-12-R1-f5-padding/README.md) |
| F6 | ResNet50 全局2048（保留，非当前优先） | `F6_ResNet50_Global2048` | 待 R3 |
| R1 | F1–F4 × LR/SVM/SupCon 十折；F5 为同运行名、同代码与协议的补充 | 运行名 `Main_Experiments`（规则前命名，不改）；存档 [2026-10-12-main-experiments](experiments/2026-10-12-main-experiments/README.md)（不改名） | 已完成并归档 |
| R2 | F3、F4、F4.1、F4.2（局部特征方案 A、B） | 实际回传运行名 `ResNet_LocalFeature_Comparison`；计划名及 `.run` 为 `R2_LocalFeatureAB`，不改已有结果名 | 已完成并[归档](experiments/2026-10-13-R2-local-features/README.md)，记录设备为 CPU |
| R1/R2 后处理 | 95%校准目标改为90%，不重新训练 | 保留两个原运行名，输出 `results_recall90` | 已核对并[归档](experiments/2026-10-13-R1-R2-recall90/README.md)，不占新训练轮次 |
| R3 | F3、F6（ResNet50 小试） | 运行名 `R3_ResNet50Pilot`；`.run` 配置 `02 R3 ResNet50 pilot (F3,F6)` | 保留，未排期 |
| R2 统计后处理 | Lecture 15公式，95%主分析及90%补充 | [2026-10-13-course-statistics](experiments/2026-10-13-course-statistics/README.md) | 已复算并独立数值核验；未重训，不占新轮次 |

新增编号或轮次时，在同一次提交中更新本表、README 和当前方案。
