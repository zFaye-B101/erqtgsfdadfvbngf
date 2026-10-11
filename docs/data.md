# SOP 数据下载与资格

## 当前入口与目录更新（2026-10-12）

- 正式入口保持 01 → 02；已有处理数据无需重跑 01。01 仅改工具导入路径和默认配置文件名。
- 总配置为 `PythonCode/config.json`；各组配置为 `PythonCode/experiment_configs/*.json`；工具在 `PythonCode/utils/`。
- 直接运行 02 会显示实验/模型菜单；命令行示例：`python 02_SOP_Experiments.py --group F1,F2 --method baselines`。
- JSON 自动发现任意新增组；已有提取器可通过 components 组合，全新方法还需实现并注册。
- 默认四组为 HSV48、HSV+分区域梯度192、ResNet全局512、ResNet全局+区域2560，均带填边。
- 结果保存至 `results/<run_name>/<group>/`；`comparisons/`读取兼容已完成结果生成逐折比较，不重复训练。
- 移动代码改变指纹，新协议目录为 `protocols/Main_Experiments_Organized`；旧输出保留，不强行混用。
- 保留组员 00 预检、共享 PyCharm 配置及历史 Notebook；运行配置默认选 F1。
- 不改变三模型、10/3商品折、校准和扰动规则；正式SOP结果及预训练ResNet运行仍未验证。


## 最新 v3 修复（2026-10-11，覆盖此前 v2 的对应状态）

- 新增仅内容区域的 HSV/HOG 提取器 `hsv_hog_content`。2026-10-12 目录重构后，旧的 handcrafted_content / handcrafted_padding_audit 配置不再是 02 入口；白边对照改用组 `F5_Traditional_Content_HSV_Gradient`（ContentHSV48、ContentHSV_HOG192），与 F1/F2 按外层折配对比较。ResNet 仍使用明确记录的填边输入。真实SOP收益未量化。
- 每个外层训练商品集合再独立预留20%校准商品（每大类至少2个，保留足够拟合商品）。校准商品不参与内层选参、归一化或训练；最终模型在其上定阈值后不再重训，直接评估外层测试。有限样本校准不保证测试召回率达到95%。
- SupCon温度边界胜出时按内层验证扩展三点窗口，最多两次、因子2、温度限制0.005至1.0；复用已评估候选、保存搜索历史并保留未解决的边界警告。
- 官方holdout默认关闭。新协议/缓存/结果使用v3目录，不能复用v2阈值或断点；已有01处理图片可继续使用。入口整理为01预处理和02统一实验命令（--method baselines|supcon），选特征和参数仍改配置。
- Windows CPU合成检查覆盖填边探针、阈值校准隔离、温度扩展和端到端流程，验证边界见[VALIDATION](../PythonCode/VALIDATION.txt)。真实SOP十折、正式holdout、预训练ResNet及GPU仍未验证。下方此前状态保留为历史记录。


## 当前 PythonCode 数据接口（2026-10-11）

新01仅完成原图资格检查、单图商品排除、224x224填边RGB JPEG和审计清单导出。统一实验入口02通过data_dir读取清单及图片，按配置重新构建共同干净/扰动特征缓存；原NPZ不再读取，因此已有合格处理目录可复用而无需重跑01。所有条件从同一JPEG解码出发；原handcrafted及ResNet使用填边图像，handcrafted_content在扰动后取内容区域计算HSV/HOG。实现了对照不等于已证明填边无影响。缓存、共同配对协议和模型输出留在本机。下文Notebook的NPZ说明仅适用于历史入口。02在`231f2d6`缺少main调用的问题已在本轮修复，并通过Mac真实子进程合成smoke；见[入口互评](reviews/2026-10-11_231f2d6_entrypoint_codex.md)。

## 数据处理的三个不同目的

依据 [Project Sections 2.b-c](../materials/official/Project.md)，不能用“已统一尺寸”代替特征设计，也不能用放大或填边补足原图资格。

| 环节 | 当前项目中的作用 | 需要保留的证据 |
| --- | --- | --- |
| 资格与基础处理 | 原图宽高检查、可读性、商品图片数、RGB/JPEG/224x224输入 | 筛选前后图片/商品数、原始尺寸、排除原因、官方split |
| 特征构建 | HSV颜色、空间梯度，或冻结CNN的整体/区域表示；LR/SVM进一步构造图片对绝对差 | 特征定义/维数、归一化、权重来源、填边是否参与；所有学习型处理的拟合范围 |
| 鲁棒性评价 | 对query加入遮挡、噪声、模糊、旋转，reference保持协议规定的状态 | 相同图片对、种子、干扰参数/区域/填充值、固定校准阈值 |

测试扰动（test-time corruption）不等于训练增强（data augmentation）。当前SupCon使用同商品不同干净图片作正例；没有实际添加的训练增强不能写入Methods。只做局部特征拼接也不能声称已经实现遮挡检测或语义部件定位。

先做适度的数据审计以支撑设计：商品图片数、大类分布、长宽比/填边比例和少量匹配失败例。选择负例时同大类不等于外观相似；现有colour_hard按内容区域HSV选择，须与随机负例分别报告，不能把它当无偏的全部困难商品。正式报告样例应在冻结评价后用于解释，不反过来调参。


数据集不在仓库内，也不应放入最终提交包。

- [作者的数据集入口](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16#stanford-online-products-dataset)
- [作者提供的下载链接](https://drive.google.com/uc?export=download&id=1TclrpQOF_ullUP99wk_gjGN8pKvtErG8)
- [数据资格核验记录](../materials/research/stanford-online-products-audit.md)

已核验原包：3,083,860,082 bytes。

```text
SHA-256 b04fa78210e69fab050ce73939550e1cd0f96890bdd98ec9fee732c3b756ef48
```

原包包含 120053 张图片、22634 个商品。120026 张原图宽高均达到 100；27 张不足，应排除而不是放大来达标。筛后 22630 个商品有至少两张图，4 个只剩一张，不能自行构成正例。

`notebooks/01_SOP_Preprocessing.ipynb` 直接读取 ZIP，无需先解压。它检查索引图片的存在性、可读性和原始尺寸，删除不合格图片及清理后仅剩一张图片的商品，再保留官方 train/test 标记并导出 224×224 RGB 图片。它另计算 48 维 HSV 颜色直方图和 144 维空间 HOG 特征；`notebooks/sop_features.py` 还提供 SIFT 图片对特征函数供后续实验调用。这些是现有代码产物，不代表最终实验特征比较方案已经锁定。

（历史 Notebook 版本）运行时会把图片、CSV 审计/清单和 `.npz` 特征保存到当前工作目录下的 `processed_sop/`；当前 `PythonCode/01_SOP_Preprocessing.py` 不生成 `.npz`，见本文上方。该目录及 ZIP/缓存由 `.gitignore` 排除。不要提交原图、数据 ZIP、完整派生特征缓存或 Notebook 内嵌样例图片；文本统计和核验结果可以共享。

如作者链接失效，应回作者入口核对，不把未经核验的第三方镜像当作同一个数据包。
