# 协作与运行路径

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

- 新增 handcrafted_content 和 handcrafted_padding_audit，支持排除填边的HSV/HOG以及与原填边版本的配对对照；原handcrafted保持为对照，ResNet仍使用明确记录的填边输入。真实SOP收益未量化。
- 每个外层训练商品集合再独立预留20%校准商品（每大类至少2个，保留足够拟合商品）。校准商品不参与内层选参、归一化或训练；最终模型在其上定阈值后不再重训，直接评估外层测试。有限样本校准不保证测试召回率达到95%。
- SupCon温度边界胜出时按内层验证扩展三点窗口，最多两次、因子2、温度限制0.005至1.0；复用已评估候选、保存搜索历史并保留未解决的边界警告。
- 官方holdout默认关闭。新协议/缓存/结果使用v3目录，不能复用v2阈值或断点；已有01处理图片可继续使用。入口整理为01预处理和02统一实验命令（--method baselines|supcon），选特征和参数仍改配置。
- Windows CPU合成检查覆盖填边探针、阈值校准隔离、温度扩展和端到端流程，验证边界见[VALIDATION](../PythonCode/VALIDATION.txt)。真实SOP十折、正式holdout、预训练ResNet及GPU仍未验证。下方此前状态保留为历史记录。


## 最新共享运行接口（2026-10-11）

PythonCode现包含01预处理入口、02统一实验入口、config.json和共同辅助模块。通过02的--method选择基线或SupCon；切换特征改配置。新固定提取器注册于sop_feature_bank.py。两种方法共享商品折/配对和像素输入。依赖、缓存、模型、路径与续跑规则见[英文说明](../PythonCode/Readme.txt)和[中文说明](../PythonCode/Readme_zh.txt)。根目录原有Notebook、数据工具和已归档开发结果保持原状。不要将机器环境、生成配对/缓存、模型权重或原图推送。


每次开始前读 [AGENTS.md](../AGENTS.md) 和 [CONTRIBUTING.md](../CONTRIBUTING.md)。每次更新共享文件，必须在同一提交/PR内同步根目录 README 及受影响的相关文件；交接时说明验证结果和未完成项。

## 同步范围

- 新增 `PythonCode/` 作为纯Python运行入口；先01再02，两个模块和英文Readme一起保留。脚本默认数据/结果路径相对自身目录，也可通过命令行参数指定；本机数据和运行产物不共享。传统基线尚无真实训练结果，现有预处理互评问题仍待处理。
- 当前仓库地址为 [zFaye-B101/erqtgsfdadfvbngf](https://github.com/zFaye-B101/erqtgsfdadfvbngf)，2026-10-11 核验为 Public，任何人都可下载；推送仍需要相应写入权限。
- Canvas 链接仅供来源追溯，可能需要学校登录；已归档原件不依赖 Canvas 登录。
- SOP 数据另从作者渠道下载。每台机器把 ZIP 放在脚本目录/父目录，或用 `--zip-path` 等命令行参数指定，不把本机绝对路径写入共享代码。
- 不共享 GitHub token、密码、`.env`、私钥或个人服务配置。

## 多人修改

1. 默认所有组员直接更新 `main`；工作区干净时开始前先 `git pull --ff-only origin main`。
2. 不要求每人开分支或 PR，按文件分工，避免同时改同一个脚本或模块。
3. 按数据/特征、基线/评估、SupCon 分工；具体文件接口需共同确认后再实现。
4. 提交前确认脚本运行产物未加入；改过 Notebook 时清除输出，检查是否误加入原图、压缩包、缓存或模型权重。
5. 代码与 README/相关文件放在同一次提交。提交后工作区干净时 `git pull --rebase origin main`，解决冲突再 `git push origin main`；不 force push，不覆盖队友改动。
6. 交接时注明改动及运行验证状态；仅写好代码不能记为测试通过。提交形式详见 [submission.md](submission.md)，共享与最终运行代码以 `PythonCode/` 中的 `.py` 为准。

配置入口见 [environment.md](environment.md)，运行与路径见 [quickstart.md](quickstart.md)。`requirements-base.txt`包含预处理和Notebook依赖，`requirements-experiments.txt`增加开发诊断和绘图；torch/torchvision按硬件另装，最终训练版本仍待验证。不要混装GUI/headless OpenCV、不共享`.venv`，交接实际解释器、版本和验收结果。`tools/requirements-docs.txt`仅供文档转换。

## 交叉评审

后续成员和 AI 可以相互检查已提交的更新，并提出不同解决方法，详见 [互评规范与模板](peer-review.md)。每次接手只先检查自己将使用或修改的模块，不做无范围的全仓库重写。

更新者交接提交号、影响文件/接口、实际验证和待办；评审者记录具体问题及证据，并比较保留现状、最小修复和有实际价值的替代方案。负责人回应后再实施，协议改变先由用户确认。记录分别区分建议、决定、代码修改和验证通过，不能仅因已推送就关闭问题。

默认只读评审实现，评审记录可按现有 main 流程提交并同步 README；不直接覆盖他人工作，也不要求强制 PR 或阻塞每次提交。文档评审与讨论可用中文，实际提交代码仍遵守 [英文代码规范](code-style.md)。此处没有启动自动监控或跨会话消息任务。

## 运行入口

正式入口是 [PythonCode/](../PythonCode/) 的编号脚本：新机器先运行只读预检 `00_Check_Environment.py`，再运行 `01_SOP_Preprocessing.py`，再按需运行 `02_SOP_Experiments.py --method baselines` 或 `--method supcon`。详见[英文说明](../PythonCode/Readme.txt)和[中文说明](../PythonCode/Readme_zh.txt)。Windows PyCharm 可直接使用仓库 `.run/` 中共享的运行配置（见 [quickstart](quickstart.md)）；修改这些配置时同步 `tests/test_pycharm_run_configs.py`。以下 Notebook 保留为历史流程/可选预览：

`notebooks/00_setup_and_data.ipynb` 用于环境检查和前 20 张图片预览，可选运行。`notebooks/01_SOP_Preprocessing.ipynb` 才执行全量 ZIP 图片审计、筛选、224×224 导出和 HSV/HOG 特征缓存；需与 `notebooks/sop_features.py` 一起使用。输出位于本机 `processed_sop/`，已由 `.gitignore` 排除。`materials/sop_data_check.ipynb` 保留为早期核验记录。Windows 全量预处理仍待运行验证；目前没有模型训练 Notebook。

仓库原为私有，现由用户改为 Public。未来任何权限或可见性变更仍需单独确认，不由 AI 擅自调整；现有资料公开不等于获得再分发授权。
