# COMP90051 Product Matching

小组协作仓库：同步作业要求、代码、研究资料与实验决策。当前地址为 [zFaye-B101/erqtgsfdadfvbngf](https://github.com/zFaye-B101/erqtgsfdadfvbngf)。2026-10-11 核验为 Public，任何人都可读取；公开不代表课程资料或论文获得了自由转载许可。不要上传数据集、凭据、个人信息或模型缓存。

## 所有协作者与 AI 必读

开始修改前，先读 [AGENTS.md](AGENTS.md) 与 [CONTRIBUTING.md](CONTRIBUTING.md)，并简短确认本次范围。**每次更新共享文件，必须在同一次提交/PR中更新本 README，以及受影响的文档、配置、调用代码、测试和资料索引。** 不支持自动读取 AGENTS.md 的 AI，也必须先由使用者提供该文件。

### 最近更新：2026-10-12

- **Windows + PyCharm 预配置（真实测试暂缓）：** 新增只读预检 [00_Check_Environment.py](PythonCode/00_Check_Environment.py)，检查解释器、依赖、PyTorch 是否为 CPU 版、数据或 ZIP 位置、磁盘空间、写权限，以及 Windows 路径长度和控制台编码；不安装、不下载。新增仓库根目录 [.run/](.run/) 共享运行配置（00 预检、01 预处理、02 两种 smoke、02 两种正式运行、单元测试）。说明见 [quickstart](docs/quickstart.md) 和 [Readme.txt](PythonCode/Readme.txt)。
- **验证：** Mac/Python 3.11 CPU 临时环境中 `PythonCode/tests` 12 项（新增 2 项预检测试）和根 `tests/` 9 项（新增运行配置检查）通过。预检的 Windows 专属分支、PyCharm 实际加载这些配置、Windows 运行、真实 SOP 均**未验证**。
- **互评复查：** [可配置流程互评](docs/reviews/2026-10-11_88e1760_pipeline_claude.md) R1–R3 改为已验证、R4 大部分已实施；[预处理互评](docs/reviews/2026-10-11_3054cd3_preprocessing-features_claude.md) R1 改为部分实施、待真实数据量化。[docs/data.md](docs/data.md) 中 `.npz` 描述标注为历史 Notebook 版本。
- 本次不改模型、协议或默认配置。更早的变更说明见 Git 历史、[docs/reviews/](docs/reviews/) 与 [当前方案](docs/current-plan.md)。

## 现在开始运行

02入口已通过命令行帮助及两种方法的合成smoke检查；正式运行仍需使用完整依赖环境和合格真实数据，不能只凭正常退出判断完成。

纯 Python 入口：在 `PythonCode/` 中使用同一个 Python 3.11+ 环境，先运行只读预检 `python 00_Check_Environment.py`，再运行 `python 01_SOP_Preprocessing.py`，再运行 `python 02_SOP_Experiments.py --method baselines` 或 `python 02_SOP_Experiments.py --method supcon`。已有完整处理结果时，设置 `data_dir` 后可跳过01。更换已有特征时修改 `experiment_config.json` 的 `feature_profile`，再运行所需方法。输出默认位于 `PythonCode/results_v3/`，按运行名、特征和算法分开保存。开发检查可加 `--smoke-test`；传统和SupCon使用相同的共同配置进行比较。

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
- 当前阶段：PythonCode v3 已实现 01 预处理与 02 统一实验入口（`--method baselines|supcon`），含独立商品校准、内容区域特征/白边对照、SupCon 有界温度扩展，官方 holdout 默认关闭。合成测试在 Windows CPU（组员记录）与 Mac（本仓库复查）通过；真实测试暂缓，之后在 Windows PyCharm 运行，已预置 00 预检与 `.run/` 配置。尚无真实 SOP 正式三模型或十折结果，最终特征和正式参数仍待验证。
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
| `.run/` | Windows PyCharm 共享运行配置（00/01/02/测试），开发便利，不进入提交包 |
| `PythonCode/` | 00只读预检、01预处理及02统一实验入口、特征/协议/模型模块、配置、合成检查和中英文说明；尚无真实正式训练结果 |
| `tests/` | 使用合成数据的数据读取测试，以及 `.run/` 配置一致性检查 |

最终提交需单独整理四页 PDF 报告，以及包含代码、`Readme.txt` 和必要自动化脚本的 ZIP；不含数据。具体格式以官方原件为准。本仓库不配置公共许可证，也不自动向 Canvas 提交。
