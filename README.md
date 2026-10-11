# COMP90051 Product Matching

小组协作仓库：同步作业要求、代码、研究资料与实验决策。当前地址为 [zFaye-B101/erqtgsfdadfvbngf](https://github.com/zFaye-B101/erqtgsfdadfvbngf)。2026-10-11 核验为 Public，任何人都可读取；公开不代表课程资料或论文获得了自由转载许可。不要上传数据集、凭据、个人信息或模型缓存。

## 所有协作者与 AI 必读

开始修改前，先读 [AGENTS.md](AGENTS.md) 与 [CONTRIBUTING.md](CONTRIBUTING.md)，并简短确认本次范围。**每次更新共享文件，必须在同一次提交/PR中更新本 README，以及受影响的文档、配置、调用代码、测试和资料索引。** 不支持自动读取 AGENTS.md 的 AI，也必须先由使用者提供该文件。

### 最近更新：2026-10-11

- **互评更新（Claude）：** 在本会话 Mac/Python 3.11 CPU 临时环境复现 `PythonCode/tests` 5 项合成测试通过（与组员 Windows 记录的库版本一致）。[预处理互评](docs/reviews/2026-10-11_3054cd3_preprocessing-features_claude.md) R2 改为已验证、R1 暂缓；新增 [可配置流程互评](docs/reviews/2026-10-11_88e1760_pipeline_claude.md)：LR/SVM 图片对特征只有绝对差，与 Readme 描述不符（R1），官方 holdout 查询覆盖缺少断言（R2），待负责人回应。只改文档，未改代码；真实 SOP、Windows、GPU 仍未验证。
**最新代码交接：固定 01/02/03 入口的可配置流程已合入 PythonCode。**

- 01 仅清理和导出图片；02 运行 LR/SVM；新增 03 运行 SupCon-based matching。配置选择 HSV48/HSV_HOG192、ResNet18 Global512 或 Global512+Local2048，无需修改编号入口。
- 02/03 共用商品折、带校验的图片对、HSV 困难负例和查询图扰动；所有特征均从同一保存 JPEG 解码图提取，修复旧流程干净 NPZ 与扰动 JPEG 输入不一致的问题。白色填边仍保留，影响尚待量化。
- SupCon 实现论文 Eq.(2)，采用固定描述子上的可训练表示网络与投影头、同商品不同干净图片作正例、保留表示网络作余弦匹配。是项目应用适配，不是原论文端到端分类复现。
- Windows CPU 上 5 项合成检查通过，含 01 筛选、两种特征的 LR/SVM/SupCon 两层折流程、五种测试条件、512 维输入、模型重载和断点复用。真实 SOP 十折、真实官方 holdout、预训练 ResNet 与 GPU 性能尚未验证。
- 使用方式见 [PythonCode/Readme.txt](PythonCode/Readme.txt)，配置见 [experiment_config.json](PythonCode/experiment_config.json)，验证边界见 [VALIDATION.txt](PythonCode/VALIDATION.txt)，历史评审回应见 [本轮交接](docs/reviews/2026-10-11-configurable-pipeline-response.md)。下方旧记录中的“SupCon尚缺/CNN尚未接入”仅描述此前版本；旧开发数值没有被重算或提升为正式结果。


**本轮交接必读：拉取后先读更新文档，再运行或修改代码。** 请组员及各自AI依次读取 [当前方案](docs/current-plan.md)、[CNN+LR三轮结果](docs/experiments/2026-10-11-lr-feature-pilot.md)、[PythonCode进度与互评](docs/reviews/2026-10-11_009a241_python-progress_codex.md)，记录所读提交和负责事项。`git fetch`不更新本地工作树；有未提交改动时先保留，不强制覆盖。本条是提醒，不表示成员已读或已同意。

- 已对齐最新PythonCode传统流程与用户CNN小试：传统LR/SVM及10x3验证代码已上传但无完整运行凭证；CNN的Global512/G+L2560已跑LR开发测试，尚未接入正式.py。SupCon尚缺。新增24行两轮选参结果、48行调参记录和12行参数/阈值；未改组员代码。
- 同配对扩大C范围后，遮挡条件G+L的平均FPR为0.2639（Global 0.3194），平均TPR均为0.7583，AUC差仅+0.0059；是三种子开发信号，不是局部稳定胜出或正式结果。具体边界见上方记录。
- 复查JPEG缓存混用和填边问题，补充内层阈值移植风险；开发传统流程建议先跳过官方holdout。CNN原图小试不必等待传统预处理修复。4个.py语法、100组指标探针及8项数据测试通过，不等于完整基线/Windows验证。

- **规范更新：以 `.py` 为正式格式。** 最终运行电脑更适合 `.py`，共享运行代码和最终提交代码以 [PythonCode/](PythonCode/) 的编号脚本为准；Notebook 保留为历史流程/可选预览，不删除。同步 AGENTS、CONTRIBUTING、[代码规范](docs/code-style.md)、[提交说明](docs/submission.md)、协作/运行/环境/互评说明。
- 复查 [预处理互评](docs/reviews/2026-10-11_3054cd3_preprocessing-features_claude.md) 在 PythonCode 中的状态：R1 填边仍在；R2 已实际触发——02 的干净条件用 `.npz` 特征、干扰条件从 JPEG 重新提取，鲁棒性差异混有压缩差异。待负责人回应，未改代码。
- 上传本机原版 [02传统基线Notebook](notebooks/02_SOP_Traditional_Baselines.ipynb) 与完整 `notebooks/sop_baselines.py`，无执行输出；本机01与已有01逐字节一致，未重复更新。按用户要求，`notebooks/` 和 `PythonCode/` 独立运行，Notebook使用同目录辅助模块，不跨目录调用。原版Notebook保留中文学习说明，不视为最终英文提交版；仅上传，未执行实验。
- 新增 [PythonCode/](PythonCode/)：上传本机 `Code/` 中全部五个英文文件，提供独立 Python 脚本入口。先运行 `01_SOP_Preprocessing.py`，再运行 `02_SOP_Traditional_Baselines.py`；无需 Notebook，详见 [英文运行说明](PythonCode/Readme.txt)。第二步包含 LR/SVM、48/192维对照、手写10×3嵌套验证和遮挡/噪声/模糊/旋转测试；尚未执行真实训练，也未实现 SupCon。原 Notebook 与组员文件保留，已有填边和JPEG缓存一致性评审仍待处理，不能将上传等同于正式实验完成。
- 新增 [预处理填边与特征缓存互评](docs/reviews/2026-10-11_3054cd3_preprocessing-features_claude.md)：R1 `resize_for_model` 的白色填边进入HSV/HOG特征，使特征携带长宽比（合成探针复现），已归档的传统特征AUC可能因此偏低；R2 `.npz` 特征按未压缩图计算，与保存的JPEG不一致。待负责人回应，未改代码；同步互评索引、诊断说明与当前方案，并在当前状态注明仍处于特征选择阶段。
- 新增 [环境配置与验收](docs/environment.md) 和 `requirements-experiments.txt`：提供Mac/Linux、Windows CPU、Windows CUDA选择流程、Notebook内核及验收代码；区分Mac实测和待验证新环境，避免混装OpenCV。未改实验实现或自动安装依赖。
- 归档 [已测特征开发诊断](docs/experiments/2026-10-11-feature-diagnostics.md)、[CNN三种子CSV](docs/experiments/2026-10-11-resnet18-seed-results.csv)、[平均/最大池化CSV](docs/experiments/2026-10-11-pooling-seed-results.csv)和[layer3/layer4 CSV](docs/experiments/2026-10-11-layer-seed-results.csv)：平均池化本轮较高；层级比较无稳定赢家。各轮独立开发抽样，不混算；不是正式模型实验。
- 新增 [layer3/layer4三种子CSV](docs/experiments/2026-10-11-layer-seed-results.csv)：三种子胜负不一致、无稳定层级赢家；更新测试安排，先回应预处理填边/特征缓存互评，再开展统一输入下的训练/成本小试。代码未改，仍属开发诊断。
- 更新 [SML课内测试安排](docs/reviews/2026-10-11_ae5a1d6_feature-options_codex.md)：池化与层级初试完成，暂无最终特征决定；暂缓DINOv2，不增加其代码或依赖，继续使用SupCon论文。
- 澄清 [文献综述要求](docs/reviews/2026-10-11_9291720_course-requirements_codex.md)：解释新方法与课堂内容的区别，不是禁止引用课堂论文；需数据集引用加至少两篇额外研究论文。
- 新增 [课程要求与staff反馈互评](docs/reviews/2026-10-11_9291720_course-requirements_codex.md)：区分官方要求、课件依据及待讨论方案；记录老师建议扩展噪声/模糊/旋转、TabM反馈适用边界，以及正式指标/CV/调参须自行实现。只改文档，未实施或批准最终方案。
- 新增 [特征方案互评](docs/reviews/2026-10-11_8990473_features_codex.md)，供组员及其AI讨论：建议小样本比较冻结ResNet18全局与2x2区域表示，HSV/HOG作为轻量对照、SIFT可选。当前仅建议，尚未定案或修改实现。
- 新增 [互评与替代方案流程](docs/peer-review.md)：按提交和代码位置记录问题、证据、备选方法、负责人回应及验证状态；评审不直接覆盖他人实现，不改变直接协作 `main` 的方式。
- 新增 `notebooks/01_SOP_Preprocessing.ipynb` 与 `notebooks/sop_features.py`：提供全量 ZIP 清理、224×224 图片导出和 HSV/HOG 特征缓存；运行结果留在本机并由 `.gitignore` 排除。代码尚未在 Windows 上完成全量运行。
- `00_setup_and_data.ipynb` 仍用于环境检查和少量图片预览；完整预处理入口是 `01_SOP_Preprocessing.ipynb`。本次只增加数据准备，不实现模型训练或正式评估。
- 此前的协作规范更新已通过文件链接/哈希检查及已有数据工具回归测试；这些结果不代表本次新增 Notebook 已在 Windows 或全量数据上验证。
- 新增 [提交代码与中文学习副本规范](docs/code-style.md)：提交代码按功能大块添加英文 Markdown，源码添加必要英文注释，自定义输出使用英文。
- 范围仅限最终提交代码及配套运行说明；资料、论文、历史文件和仓库管理文档不要求翻译。可另建含对应源码、中文注释和逐块详解的学习副本，不进入最终提交包。
- 英文实现为唯一依据，已有中文副本随源码同步；继续直接协作 `main`，每批共享改动同步 README 与相关文件。
- 本次补充环境配置和验收入口；已保留诊断结果文档/CSV和测试安排，未改实验实现、重跑试验或启动监控。早期互评保留为历史快照，开发诊断不等于正式实验完成。

## 现在开始运行

纯 Python 入口：在 `PythonCode/` 中使用同一个 Python 3.11+ 环境，依次运行 `python 01_SOP_Preprocessing.py`、`python 02_SOP_Traditional_Baselines.py`、`python 03_SOP_SupCon.py`。已有完整处理结果时，配置 `data_dir` 后跳过01。以后换特征主要修改 `experiment_config.json` 的 `feature_profile`，重新运行02/03。输出默认位于 `PythonCode/results_v2/`，按运行名、特征和算法分开保存。开发先使用 `--smoke-test`，或02/03同时加 `--skip-official-test`；正式设置与历史CNN小试仍须区分。

详细依赖、数据路径、缓存与运行方法见 [英文运行说明](PythonCode/Readme.txt) 和 [quickstart](docs/quickstart.md)。数据、缓存、模型权重和机器环境留在本机。

## 先读这些

1. [官方作业全文（Markdown）](materials/official/Project.md) / [PDF 原件](materials/official/Project.pdf)
2. [官方要求核对清单](materials/official/REQUIREMENTS.md)
3. [当前研究方向与待定事项](docs/current-plan.md)
4. [协作与本地路径](docs/collaboration.md)
5. [数据下载与已完成核验](docs/data.md)
6. [全部资料清单及转换说明](docs/materials.md)
7. [互评、替代方案与记录模板](docs/peer-review.md)

## 当前状态：2026-10-11

- 研究问题：面对训练时未见过的商品，局部视觉特征是否能减少部分遮挡下相似商品的误匹配，同时维持同商品匹配率？
- 当前阶段：CNN特征LR开发选参已完成三轮；可配置的共同特征/配对接口及LR、SVM、SupCon已实现并通过Windows CPU合成检查，尚未完成真实SOP正式三模型或十折实验。最终特征和正式参数仍待验证。
- 当前算法方向：Logistic Regression、RBF SVM、SupCon-based matching。不是旧草稿中的 LightGBM 或 TabM。
- 代码格式：共享与最终提交代码以 `PythonCode/` 的 `.py` 为准，Notebook 不再作为正式入口。
- SOP 数据资格检查已完成；`PythonCode/01_SOP_Preprocessing.py`（及历史 Notebook 版本）已提供全量预处理代码，但 Windows 全量运行尚未验证，也不是已完成的模型或实验。
- 用户确认 Windows/PyCharm 主机已下载数据；新公共读取代码在 Mac 上通过测试，Windows 依赖与 GPU 尚待运行检查。
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
| `docs/experiments/` | 用户回传的开发诊断与证据边界；不含数据、缓存或正式训练结果 |
| `tools/` | 文档提取工具，不是实验代码 |
| `sop_project/` | 可复用的环境诊断、SOP 数据读取和核验 |
| `notebooks/` | 历史/可选的独立Notebook流程，非正式提交入口：环境预览、预处理、传统基线02及同目录辅助模块；尚无正式训练结果 |
| `PythonCode/` | 固定01/02/03英文入口、特征/协议/模型辅助模块、配置、合成测试和说明；已做Windows CPU合成验证，尚无真实正式训练结果 |
| `tests/` | 使用合成数据的数据读取测试 |

最终提交需单独整理四页 PDF 报告，以及包含代码、`Readme.txt` 和必要自动化脚本的 ZIP；不含数据。具体格式以官方原件为准。本仓库不配置公共许可证，也不自动向 Canvas 提交。
