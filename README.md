# COMP90051 Product Matching

小组协作仓库：同步作业要求、代码、研究资料与实验决策。当前地址为 [zFaye-B101/erqtgsfdadfvbngf](https://github.com/zFaye-B101/erqtgsfdadfvbngf)。2026-10-11 核验为 Public，任何人都可读取；公开不代表课程资料或论文获得了自由转载许可。不要上传数据集、凭据、个人信息或模型缓存。

## 所有协作者与 AI 必读

开始修改前，先读 [AGENTS.md](AGENTS.md) 与 [CONTRIBUTING.md](CONTRIBUTING.md)，并简短确认本次范围。**每次更新共享文件，必须在同一次提交/PR中更新本 README，以及受影响的文档、配置、调用代码、测试和资料索引。** 不支持自动读取 AGENTS.md 的 AI，也必须先由使用者提供该文件。

### 最近更新：2026-10-12

- **Methods 小标题（用户确认）：** 在[报告提纲](docs/report-outline.md)中将数据集与预处理、特征与实验组、算法与超参数、交叉验证/扰动条件/指标整理为四个小节；说明三类选择的设计依据写入 Methods，选参结果和指标表现留给 Results and Discussion。仅更新报告结构，不改实验协议或代码。
- **报告绘图脚本：** 新增 [tools/plot_report_figures.py](tools/plot_report_figures.py)，只读归档 CSV 生成报告图 1（ROC-AUC 随条件变化）与图 2（逐折配对差值，含 TPR −1 个百分点界线）；R1 版 PDF/PNG 存于 `docs/report-figures/`，R2 存档后用同一脚本加参数重新生成。提纲第 4、6 节同步。
- **研究问题定稿（用户确认）：** 对训练中未见过的商品，加入局部视觉特征，能否在遮挡、噪声、模糊和旋转条件下减少外观相似商品之间的误匹配，同时保持同商品匹配能力？与 current-plan 英文版一致，实验与 R2 判断标准不变；README 当前状态与报告提纲同步为该措辞，并在提纲第 6 节说明 F1→F2 也可作为手工表示中“加入局部特征”的证据及其限制。
- **报告分工：** [报告提纲](docs/report-outline.md)第 6 节记录用户提供的分工（A 方法、B 结果与讨论、C 引言/文献/结论/合并排版）及进度表，并列出与提纲的待确认差异：结果部分重点、F4.1/F4.2 与 R2 未分配、绘图脚本/代码 ZIP/会议记录未分配。
- **报告提纲：** 新增 [docs/report-outline.md](docs/report-outline.md)，作为四页报告的共同提纲：总体意见与主线、评分标准与课程要求对照、各节要点/引用/必须披露事项/页数参考（只控制细节程度）、R1 与 F5 可直接使用的结果表（由已归档 CSV 生成）、图表计划、写作前核对清单、分工表，以及每节可参考的仓库文件。原 `submission.md` 的三节报告建议和各互评/存档中的“报告须说明”条目已并入提纲（来源见其第 7 节），`submission.md` 改为指向提纲。
- **R2 判断标准与退出诊断（用户确认）：** [当前方案](docs/current-plan.md)写入运行前锁定的标准：TPR 配对差值均值下降不超过 1 个百分点，FPR 下降且至少 8/10 折一致，两者同时满足；以全部和随机负例为主，困难负例只作补充并注明对颜色特征有偏。02 结束时打印 `All requested runs finished`，并在运行目录设退出看门狗（正常退出自动删除，120 秒未退出则记录线程调用栈）；只改入口脚本，代码指纹不变。PythonCode 30 项、根 9 项测试通过。
- **R2 加入 F4.2（用户决定）：** 第 2 轮改为 F3、F4、F4.1（方案 A）、F4.2（方案 B）一起运行，运行名统一为 `R2_LocalFeatureAB`；`.run` 改为 `02 Smoke F4.1,F4.2 (plans A,B)` 与 `02 R2 Local features A,B (F3,F4,F4.1,F4.2)`。说明中“新组可直接加入 Main_Experiments”的写法改为指向 R2（R1 已结束，新旧指纹不能自动配对），并修正命名约定笔误。本地真实图片子集（1690 张）上四组 smoke 通过（CPU，8.7 分钟，主要是 F4.2 五窗口提取）。Mac MPS 问题按用户决定不修改。
- **F5 白边对照结果存档（R1 补充）：** [结果与解读](docs/experiments/2026-10-12-R1-f5-padding/README.md)：300 份预测、900 行逐折指标、720 行汇总、14400 行涉及 F5 的差值核对通过，测试图片对与 R1 相同。去掉白边没有稳定改善；困难负例的选择方式对 F5 有偏。新增核对脚本 `tools/audit_f5_archive.py`，`.gitattributes` 保留存档字节；登记表与当前方案同步。模型文件和原始 ZIP 留在本机。
- **编号规则与中文提交（用户要求）：** 新增 [编号与版本规则](docs/versioning.md)：特征组新方法用 F<n>、改进用 F<n>.<k>；正式运行按轮次 R<n> 编号，运行名 `R<n>_<主题>`，结果存档目录 `<日期>-R<n>-<主题>`；含 F/R 登记表。R1 = 已归档的 `Main_Experiments`（不改名），待运行的 `LocalFeature_A`、`ResNet50_Pilot` 改名为 `R2_LocalFeatureA`、`R3_ResNet50Pilot`（`.run` 同步）。提交说明自本次起使用中文，已写入 AGENTS 与 CONTRIBUTING。未改代码。
- **局部特征方案 A = F4.1（Claude，用户选择，当前优先）：** F3 全局 512 维 + F4 四个区域向量逐通道最大值 512 维，共 1024 维；新增组件变换 `region_max`，复用同一次 ResNet18 提取。按用户约定，新方法用新编号、已有方法的改进用子版本（F4.1、F4.2）；组和配对差值按版本排序（差值为 F4.1−F4）。代码指纹变化，按 [当前方案](docs/current-plan.md) 用运行名 `R2_LocalFeatureA`（第 2 轮）重跑 F3、F4、F4.1（`.run/` 已加两个配置）。F6 保留但非当前优先。Mac 上 24 + 9 项测试及 F3/F4/F4.1 合成端到端运行通过；真实 SOP 未运行。
- **新增 ResNet50 小试准备（Claude，用户确认）：** 新增冻结 ResNet50（`IMAGENET1K_V2`）全局 2048 维提取器与组 F6，输入与 F3 相同，用于比较骨干深度；另注册 `resnet50_regions` 供以后在同一骨干内比较全局与局部。02 与 00 的 ResNet 检测改为通用。因模型源码指纹变化，按 [当前方案](docs/current-plan.md) 用运行名 `R3_ResNet50Pilot`（第 3 轮）把 F3 与 F6 一起重跑（`.run/` 已加两个配置）。Mac/Python 3.11 CPU 上 22 + 9 项测试及 F3,F6 合成端到端运行通过；官方权重与真实 SOP 未运行。
- **局部特征互评与待确认测试：** 新增[局部特征方案和测试安排](docs/reviews/2026-10-12_0cdaecc_local-feature-plan_codex.md)：建议优先跨区域最大汇总，局部裁剪分别编码及局部PCA为后续候选；明确课程依据、成本、缓存/折内拟合验收和外层评估边界。保持原研究问题，TPR/FPR必须联合判断。本会话建议暂缓骨干扩展，但并行提交3587ad1已新增F6；保留其全部实现，两条安排待团队协调，不自动取消F6。请组员与各自AI拉取后阅读并回应；三个局部候选均未实施，本批仅文档，不改代码/配置，不影响用户报告正在运行的F5。（当时状态；F5 已完成，见[F5 结果](docs/experiments/2026-10-12-R1-f5-padding/README.md)。）
- **结果补充与依赖（Claude）：** 在[结果报告](docs/experiments/2026-10-12-main-experiments/README.md)中新增“补充分析”一节（原内容未改）：F4−F3 逐折方向（如 SupCon 模糊条件困难负例 FPR 10/10 折升高、LR 遮挡 AUC 0/10 折更高），以及 95% 召回工作点 FPR 偏高、干净阈值在扰动下失准的说明；数字均由已归档 CSV 计算。`PythonCode/requirements-base.txt` 的 OpenCV 上限由 `<5` 放宽为 `<6`，注明本轮结果实际在 OpenCV 5.0.0 下产出；同步 [环境说明](docs/environment.md) 与 [当前方案](docs/current-plan.md)。未改代码，未重新训练。
- **完整真实实验回传：** 另一台Windows GPU电脑完成F1–F4的LR/SVM/SupCon十折实验。[结果与解读](docs/experiments/2026-10-12-main-experiments/README.md)包含1626份结果证据；600份预测、1800行逐折指标、1440行汇总和10800行跨组差值核对通过。ResNet整体优于传统特征，F4未稳定优于F3；F5与官方holdout未完成。用户授权发布结果，模型、数据和原始ZIP留在本地。
- **指纹保存：** 结果文件和模型/协议工具源码关闭Git换行转换，保留运行时的Windows字节格式，使已记录的结果SHA256和模型源码指纹在检出后仍一致；算法未改。
- **GPU 启动流程修正：** 02 在实验开始前加载所选方法需要的 PyTorch/torchvision，并实际执行小型设备运算检查，避免 baseline 和图表完成后才首次加载 PyTorch。模型与协议源码指纹不变。代码可传至另一台 GPU 电脑运行，本开发电脑负责代码整理与回传结果核对；组员的 F5 与分组预检更新已保留。
- **修正后的合并代码检查：** Windows CPU 上 `PythonCode/tests` 20 项（99.267秒）及根目录 `tests` 9 项（0.281秒）通过；含新进程初始化顺序、图表生成、真实CLI合成smoke和续跑。真实结果另按回传文件核对，本开发电脑没有运行正式实验。

- **预检与白边对照（Claude）：** `00_Check_Environment.py` 改为读取 `experiment_configs/` 中的组（`--group`，默认全部）：选中 ResNet 组而缺 torch 时报 FAIL，并按实际运行名、组名、特征名估算最长输出路径（原固定估计偏短）。新增 F5 组（仅内容区域的 HSV48 与 HSV_HOG192），用于和 F1/F2 做白边对照。Mac/Python 3.11 CPU 上 `PythonCode/tests` 18 项、根 `tests` 9 项通过；F2、F5 两组合成 smoke 实测最长路径与估算一致。Windows 与真实 SOP 未验证。`.run` 正式配置仍只跑 F1、`config.json` 旧 profile 未改动，待决定。
- **本次同步验证：** Windows Python 3.11 CPU 上，更新后 `PythonCode/tests` 16 项及根目录 `tests` 9 项全部通过，包含真实子进程 F1 LR/SVM 与 SupCon 合成 smoke、预检、续跑、动态组和跨组比较；`git diff --check` 通过。没有运行正式 SOP 或 PyCharm GUI，详见 [验证记录](PythonCode/VALIDATION.txt)。

## 当前入口与目录更新（2026-10-12）

- 正式入口保持 01 → 02；已有处理数据无需重跑 01。01 仅改工具导入路径和默认配置文件名。
- 总配置为 `PythonCode/config.json`；各组配置为 `PythonCode/experiment_configs/*.json`；工具在 `PythonCode/utils/`。
- 直接运行 02 会显示实验/模型菜单；命令行示例：`python 02_SOP_Experiments.py --group F1,F2 --method baselines`。
- JSON 自动发现任意新增组；已有提取器可通过 components 组合，全新方法还需实现并注册。
- 默认五组：F1 HSV48、F2 HSV+分区域梯度192、F3 ResNet全局512、F4 ResNet全局+区域2560（均带填边），F5 仅内容区域的 HSV48 与 HSV+梯度192（白边对照）。
- 结果保存至 `results/<run_name>/<group>/`；`comparisons/`读取兼容已完成结果生成逐折比较，不重复训练。
- 移动代码改变指纹，新协议目录为 `protocols/Main_Experiments_Organized`；旧输出保留，不强行混用。
- 保留组员 00 预检、共享 PyCharm 配置及历史 Notebook；运行配置默认选 F1。
- 不改变三模型、10/3商品折、校准和扰动规则；F1–F4三算法真实十折结果已回传核对，F5与官方holdout仍缺结果。



- **Windows + PyCharm 预配置（真实测试暂缓）：** 新增只读预检 [00_Check_Environment.py](PythonCode/00_Check_Environment.py)，检查解释器、依赖、PyTorch 是否为 CPU 版、数据或 ZIP 位置、磁盘空间、写权限，以及 Windows 路径长度和控制台编码；不安装、不下载。新增仓库根目录 [.run/](.run/) 共享运行配置（00 预检、01 预处理、02 两种 smoke、02 两种正式运行、单元测试）。说明见 [quickstart](docs/quickstart.md) 和 [Readme.txt](PythonCode/Readme.txt)。
- **验证：** Mac/Python 3.11 CPU 临时环境中 `PythonCode/tests` 12 项（新增 2 项预检测试）和根 `tests/` 9 项（新增运行配置检查）通过。预检的 Windows 专属分支、PyCharm 实际加载这些配置、Windows 运行、真实 SOP 均**未验证**。
- **互评复查：** [可配置流程互评](docs/reviews/2026-10-11_88e1760_pipeline_claude.md) R1–R3 改为已验证、R4 大部分已实施；[预处理互评](docs/reviews/2026-10-11_3054cd3_preprocessing-features_claude.md) R1 改为部分实施、待真实数据量化。[docs/data.md](docs/data.md) 中 `.npz` 描述标注为历史 Notebook 版本。
- 本次不改模型、协议或默认配置。更早的变更说明见 Git 历史、[docs/reviews/](docs/reviews/) 与 [当前方案](docs/current-plan.md)。

## 现在开始运行

02入口已通过命令行帮助及两种方法的合成smoke检查；正式运行仍需使用完整依赖环境和合格真实数据，不能只凭正常退出判断完成。

纯 Python 入口：可先运行只读 `00_Check_Environment.py`；未处理数据时运行 01，已有结果可跳过。运行 `02_SOP_Experiments.py` 交互选择组和模型，或用 `--group F1,F2 --method baselines`。公共设置在 `config.json`，组配置在 `experiment_configs/`，工具在 `utils/`。默认输出为 `results/<run_name>/<group>/`。

详细依赖、数据路径、缓存与运行方法见 [英文运行说明](PythonCode/Readme.txt) 和 [quickstart](docs/quickstart.md)。数据、缓存、模型权重和机器环境留在本机。

## 先读这些

1. [官方作业全文（Markdown）](materials/official/Project.md) / [PDF 原件](materials/official/Project.pdf)
2. [官方要求核对清单](materials/official/REQUIREMENTS.md)
3. [当前研究方向与待定事项](docs/current-plan.md)
4. [协作与本地路径](docs/collaboration.md)
5. [数据下载与已完成核验](docs/data.md)
6. [全部资料清单及转换说明](docs/materials.md)
7. [互评、替代方案与记录模板](docs/peer-review.md)
8. [报告提纲](docs/report-outline.md)：四页报告的分节要点、引用、图表计划、评分对照、现有结果表和各部分可参考的文件

## 当前状态：2026-10-12

- **局部方案统一：** F7与组员F4.1计算相同，删除重复F7配置，只保留F4.1组件变换实现；F8改名F4.2。A为区域最大汇总1024维，B为五窗口独立编码最大汇总1024维，F6保留。代码、说明与测试同步更新，已有结果不删除或改名。正式效果仍待GPU实验。

- 研究问题（2026-10-12 用户确认）：对训练中未见过的商品，加入局部视觉特征，能否在遮挡、噪声、模糊和旋转条件下减少外观相似商品之间的误匹配，同时保持同商品匹配能力？
- 当前阶段：可扩展01/02入口、独立商品校准、SupCon有界温度扩展与F5白边对照已实现。Windows GPU电脑已完成F1–F4三算法真实十折实验，F5白边对照结果也已单独核对归档；官方holdout未评估，完整预处理manifest和全部划分清单未回传，最终报告仍需说明验证边界。
- 当前算法方向：Logistic Regression、RBF SVM、SupCon-based matching。不是旧草稿中的 LightGBM 或 TabM。
- 代码格式：共享与最终提交代码以 `PythonCode/` 的 `.py` 为准，Notebook 不再作为正式入口。
- SOP 数据资格检查和全量预处理代码已提供；用户曾提供成功预处理截图，本次完整实验结果已回传，但未包含全量manifest，不能据此独立重新审核所有图片资格。
- 用户在另一台Windows GPU电脑运行；回传SupCon/ResNet元数据记录CUDA，开发电脑只整理代码、运行合成检查并核对结果。
- 官方 DOCX 是空白模板，不是已提交版本。历史文件中的提交状态只代表当时的记录。
- 资料归档不代表代码可复现、实验完成或作业提交。

## 内容

| 目录 | 用途 |
| --- | --- |
| `materials/official/` | 作业说明、要求、三份官方模板；原件和 Markdown 版 |
| `materials/planning/` | 早期选题和启动草稿，保留历史，不作为当前方案 |
| `materials/research/` | SOP 数据核验及旧候选研究记录 |
| `materials/guidance/` | 用户已删除 PPT 并清空说明；不恢复 |
| `materials/sop_data_check.ipynb` | 数据核验代码，已清除嵌入图片与运行输出 |
| `papers/` | 当前选定的 SupCon 论文，PDF 和 Markdown |
| `docs/` | 当前方案、协作说明、下载说明、文件清单 |
| `docs/reviews/` | 按提交定位的互评、替代方案及待回应事项；不代表已采纳 |
| `docs/experiments/` | 开发诊断及用户授权发布的F1–F4正式十折结果证据；不含图片、特征缓存或模型参数 |
| `tools/` | 文档提取工具，不是实验代码 |
| `sop_project/` | 可复用的环境诊断、SOP 数据读取和核验 |
| `notebooks/` | 历史/可选的独立Notebook流程，非正式提交入口：环境预览、预处理、传统基线02及同目录辅助模块；尚无正式训练结果 |
| `.run/` | Windows PyCharm 共享运行配置（00/01/02/测试），开发便利，不进入提交包 |
| `PythonCode/` | 00预检、01预处理及02统一入口、配置、模型/协议工具、检查和中英文说明；真实结果见docs/experiments |
| `tests/` | 使用合成数据的数据读取测试，以及 `.run/` 配置一致性检查 |

最终提交需单独整理四页 PDF 报告，以及包含代码、`Readme.txt` 和必要自动化脚本的 ZIP；不含数据。具体格式以官方原件为准。本仓库不配置公共许可证，也不自动向 Canvas 提交。
