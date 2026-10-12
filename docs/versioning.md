# 编号与版本规则

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

- 互评记录中的问题编号 R1、R2……只在该记录内部有效，与实验轮次无关；引用时写明记录文件，例如“局部特征互评 R2”。
- 提交说明（commit message）使用中文，可保留 `feat:`、`fix:`、`docs:` 等类型前缀，例如 `feat: 新增 F4.1 跨区域最大汇总`。2026-10-12 之前的英文提交不改写。

## 4. 登记表

| 编号 | 内容 | 运行名 / 位置 | 状态 |
| --- | --- | --- | --- |
| F1 | HSV48 | `F1_Traditional_HSV48` | R1 已完成 |
| F2 | HSV＋空间梯度192 | `F2_Traditional_HSV_Gradient192` | R1 已完成 |
| F3 | ResNet18 全局512 | `F3_ResNet_Global512` | R1 已完成；R2、R3 中重跑 |
| F4 | ResNet18 全局＋2×2区域2560 | `F4_ResNet_GlobalLocal2560` | R1 已完成；R2 中重跑 |
| F4.1 | 局部特征方案 A：全局512＋跨区域最大512 | `F4.1_ResNet_GlobalRegionMax1024` | 待 R2 |
| F4.2 | 局部特征方案 B：全局512＋五窗口独立编码最大512 | `F4.2_ResNet_GlobalCropMax1024` | 待 R2 |
| F5 | 仅内容区域的 HSV48/HSV＋梯度192（白边对照） | `F5_Traditional_Content_HSV_Gradient` | R1 补充已完成；存档 [2026-10-12-R1-f5-padding](experiments/2026-10-12-R1-f5-padding/README.md) |
| F6 | ResNet50 全局2048（保留，非当前优先） | `F6_ResNet50_Global2048` | 待 R3 |
| R1 | F1–F4 × LR/SVM/SupCon 十折；F5 为同运行名、同代码与协议的补充 | 运行名 `Main_Experiments`（规则前命名，不改）；存档 [2026-10-12-main-experiments](experiments/2026-10-12-main-experiments/README.md)（不改名） | 已完成并归档 |
| R2 | F3、F4、F4.1、F4.2（局部特征方案 A、B，当前优先） | 运行名 `R2_LocalFeatureAB`；`.run` 配置 `02 R2 Local features A,B (F3,F4,F4.1,F4.2)` | 待在 Windows GPU 电脑运行 |
| R3 | F3、F6（ResNet50 小试） | 运行名 `R3_ResNet50Pilot`；`.run` 配置 `02 R3 ResNet50 pilot (F3,F6)` | 保留，未排期 |

新增编号或轮次时，在同一次提交中更新本表、README 和当前方案。
