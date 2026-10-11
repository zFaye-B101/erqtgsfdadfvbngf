# COMP90051 Product Matching

小组协作仓库：同步作业要求、代码、研究资料与实验决策。当前地址为 [zFaye-B101/erqtgsfdadfvbngf](https://github.com/zFaye-B101/erqtgsfdadfvbngf)。2026-10-11 核验为 Public，任何人都可读取；公开不代表课程资料或论文获得了自由转载许可。不要上传数据集、凭据、个人信息或模型缓存。

## 所有协作者与 AI 必读

开始修改前，先读 [AGENTS.md](AGENTS.md) 与 [CONTRIBUTING.md](CONTRIBUTING.md)，并简短确认本次范围。**每次更新共享文件，必须在同一次提交/PR中更新本 README，以及受影响的文档、配置、调用代码、测试和资料索引。** 不支持自动读取 AGENTS.md 的 AI，也必须先由使用者提供该文件。

### 最近更新：2026-10-11

- 新增 [环境配置与验收](docs/environment.md) 和 `requirements-experiments.txt`：提供Mac/Linux、Windows CPU、Windows CUDA选择流程、Notebook内核及验收代码；区分Mac实测和待验证新环境，避免混装OpenCV。未改实验实现或自动安装依赖。
- 归档 [已测特征开发诊断](docs/experiments/2026-10-11-feature-diagnostics.md)及[CNN三种子CSV](docs/experiments/2026-10-11-resnet18-seed-results.csv)：用户回传结果与工作台汇总分开标注，缺失原始日志/配置不补造；非正式模型实验。
- 更新 [SML课内测试安排](docs/reviews/2026-10-11_ae5a1d6_feature-options_codex.md)：池化、中间层及现有表示的训练/成本验证；暂缓DINOv2，不增加其代码或依赖，继续使用SupCon论文。
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

先看 [拉取与运行说明](docs/quickstart.md)，按 [环境配置与验收](docs/environment.md) 配置后端与内核。可选运行 [00_setup_and_data.ipynb](notebooks/00_setup_and_data.ipynb) 预览少量图片；确定需要全量导出后才运行 [01_SOP_Preprocessing.ipynb](notebooks/01_SOP_Preprocessing.ipynb)。SOP ZIP 和生成的 `processed_sop/` 都留在本机，不上传。

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
- 当前算法方向：Logistic Regression、RBF SVM、SupCon-based matching。不是旧草稿中的 LightGBM 或 TabM。
- SOP 数据资格检查已完成；`01_SOP_Preprocessing.ipynb` 已提供全量预处理代码，但 Windows 全量运行尚未验证，也不是已完成的模型或实验。
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
| `notebooks/` | 环境/预览入口与全量数据预处理；不含模型训练 |
| `tests/` | 使用合成数据的数据读取测试 |

最终提交需单独整理四页 PDF 报告，以及包含代码、`Readme.txt` 和必要自动化脚本的 ZIP；不含数据。具体格式以官方原件为准。本仓库不配置公共许可证，也不自动向 Canvas 提交。
