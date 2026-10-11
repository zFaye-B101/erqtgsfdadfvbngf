# 从 GitHub 拉取并运行

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


## 当前实验流程

### Windows + PyCharm（最终运行机）

真实测试将在一台 Windows 电脑的 PyCharm 中运行，仓库已预置：

1. 用 PyCharm 打开**仓库根目录**（含 `.run/` 与 `PythonCode/`），在 Settings > Project > Python Interpreter 选择已装依赖的 Python 3.11 解释器。建议把项目放在较短路径（如 `D:\sml\`），结果目录层级较深。
2. Run 下拉框会出现 `.run/` 中共享的配置：`00 Check environment` → `01 Preprocess SOP` → `02 Smoke baselines` / `02 Smoke SupCon` → `02 Baselines` / `02 SupCon`，另有 `Tests PythonCode`。工作目录都是 `PythonCode/`，并设置 `PYTHONUTF8=1`。
3. 先运行 `00 Check environment`（只读，不安装不下载）。它报告解释器、依赖、PyTorch 是否为 CPU 版、数据或 ZIP 位置、磁盘空间、写权限、Windows 路径长度与控制台编码；有 FAIL 时先处理。
4. 若 PyCharm 提示找不到模块或解释器（如项目文件夹改名），在 Run > Edit Configurations 里重新选择解释器。ZIP 不在仓库旁时，在 `01 Preprocess SOP` 的参数中加 `--zip-path "D:/data/Stanford_Online_Products.zip"`。

这些配置只在 Mac 上做了文件与入口一致性检查，尚未在 Windows PyCharm 中实际加载或运行。

### 命令行

在 `PythonCode/` 中使用同一 Python 3.11+ 解释器。先运行一次预处理，再按需运行一种或两种实验方法：

```powershell
conda activate Unimelb_SML_A2
cd PythonCode
python 01_SOP_Preprocessing.py
python 02_SOP_Experiments.py --group F1 --method baselines
python 02_SOP_Experiments.py --group F1 --method supcon
```

02 的 `--method` 选择传统 LR/SVM 基线或 SupCon。已有完整处理目录时，在配置中设置 `data_dir` 后可跳过 01。更换特征时选择 experiment_configs 下对应组；使用 --group 或交互菜单。

当前菜单自动读取 `experiment_configs/` 下的 F1～F4 配置，并支持后续新增组。旧 `handcrafted` 等 profile 保留在公共配置中作为内部兼容设置，不再是菜单入口。传统与 SupCon 使用同一套商品折分、图片对和扰动条件。ResNet18 特征提取器已经实现，但真实预训练权重实验仍需单独运行验证。

基础依赖见 [requirements-base.txt](../PythonCode/requirements-base.txt)。已有 Conda OpenCV 无需再安装 pip OpenCV。SupCon 和 ResNet 特征需要兼容的 torch/torchvision，CPU 安装命令见[英文说明](../PythonCode/Readme.txt)，中文说明见[中文版](../PythonCode/Readme_zh.txt)。

01 默认在脚本目录及其父目录查找 SOP ZIP，也支持 `--zip-path`；输出默认为 `processed_sop/`，不会覆盖非空输出目录。实验输出默认按运行名、实验组和方法保存在 `results/`。02 可通过 `--data-dir`、`--results-dir`、`--protocol-dir` 指定目录，也支持 `--all-train`、`--skip-official-test` 和 `--smoke-test`。

开发检查可分别运行：

```powershell
python 02_SOP_Experiments.py --group F1 --method baselines --smoke-test
python 02_SOP_Experiments.py --group F1 --method supcon --smoke-test
```

`--smoke-test` 只检查程序执行，不构成正式实验结果。测试命令为 `python -m unittest discover -s tests -v`；本次入口合并没有重跑测试，既有验证边界见 [VALIDATION.txt](../PythonCode/VALIDATION.txt)。

## 历史 Notebook 流程

以下 Notebook 操作仅适用于历史或可选预览入口，不是当前 `.py` 实验的依赖。
## 1. 获取仓库

原版 [02_SOP_Traditional_Baselines.ipynb](../notebooks/02_SOP_Traditional_Baselines.ipynb) 也已上传，保留用户的中文学习说明；01与已有版本完全相同。完整 `sop_baselines.py` 与 `sop_features.py` 位于同一 `notebooks/` 目录，不依赖 `PythonCode/`。用户明确要求两套目录独立运行；未来改动应注明更新哪套入口，不能假设自动同步。

Notebook 01的输出相对其工作目录；02的 `DATA_DIR` 默认指向仓库根目录的 `processed_sop/`，运行前应改为01实际输出的位置。若单独下载并打开 `notebooks/`，只需保留两份Notebook与两个同目录辅助模块，并指定本机数据路径。结果默认放在02计算出的 `PROJECT_ROOT` 下的 `results_traditional_v1/`，不要混用不同入口的检查点。Notebook原版并非已整理的最终英文提交文件，上传时无执行输出，也未运行训练。

首次在 Windows 的终端运行（需要 Git；当前仓库为 Public，克隆无需登录，推送需要写入权限）：

```bash
git clone https://github.com/zFaye-B101/erqtgsfdadfvbngf.git
cd erqtgsfdadfvbngf
```

已有克隆时，在仓库目录运行；旧名字的本地文件夹可以保留：

```bash
git remote set-url origin https://github.com/zFaye-B101/erqtgsfdadfvbngf.git
git pull --ff-only
```

若提示本地修改冲突，先提交自己的修改或保留备份再处理，不使用强制覆盖命令。如果没有 Git，也可在仓库网页使用 Code > Download ZIP；但 ZIP 下载不是可持续拉取的 Git 克隆。修改前先读 [AGENTS.md](../AGENTS.md)，每次变更同时更新 README 与相关文件。

本组当前直接更新 `main`，不要求个人分支或 PR。各人负责不同文件，避免同时编辑同一个脚本或模块。自己提交完成且工作区干净后，先 `git pull --rebase origin main` 再 `git push origin main`；发生冲突先处理，不强制推送。详见 [协作规范](../CONTRIBUTING.md)。

## 2. 在 PyCharm 打开整个仓库目录

先按 [环境配置与验收](environment.md) 配置Python3.12、依赖与内核。新环境可装`requirements-experiments.txt`（包含基础依赖），torch/torchvision另按Mac、CPU或CUDA安装。已有GUI版OpenCV时不要直接叠装headless。

选择本项目的 Python 解释器，并打开 [00_setup_and_data.ipynb](../notebooks/00_setup_and_data.ipynb)。从第一个代码格开始运行。

- 第一格输出当前内核、Python、依赖导入结果与 GPU 信息，不自动安装软件。
- 缺少基础依赖时才启用安装格的 `INSTALL_BASE = True`，安装到当前内核。基础包不包含 PyTorch。
- 后续格会提示输入 SOP ZIP 或已解压文件夹的完整路径。支持直接包含 `Ebay_train.txt` / `Ebay_test.txt` 的目录，或它上一级且子文件夹叫 `Stanford_Online_Products` 的目录。
- 检查前 20 张图片并显示第一张。前 20 张只是读取测试，不是全量资格核验，也不能用于正式划分。

`00_setup_and_data.ipynb` 是可选的环境/少量图片预览，不是全量预处理的前置步骤。环境可用后，在同一 PyCharm 项目中打开 `notebooks/01_SOP_Preprocessing.ipynb`，并选择已安装 `numpy`、`pandas`、Pillow 和 OpenCV 的内核。配套模块 `notebooks/sop_features.py` 需与 Notebook 一起保留。该 Notebook 直接读取 SOP ZIP；若当前目录和父目录找不到 ZIP，按提示输入完整路径。

全量预处理会检查 ZIP 索引对应的图片是否存在、能否解码、原始宽高是否均至少 100；排除只剩一张合格图片的商品，导出保持比例并填充到 224×224 的图片，同时计算 192 维 HSV+空间 HOG 特征。默认输出到 Notebook 当前工作目录的 `processed_sop/`，并生成清单、排除记录、全量审计表和摘要。它不创建实验划分、不训练模型，也不生成遮挡样本。运行时间和磁盘占用会随完整数据集而增加。

如果当前 PyCharm 无法打开 Notebook，仍可用终端方式继续下一节，不必立即更换软件。

## 3. 终端等价操作

在仓库根目录，并使用与 PyCharm 项目一致的解释器：

```bash
python -m sop_project.environment
python -m pip install -r requirements-base.txt
python -m sop_project.data --data "D:/datasets/Stanford_Online_Products.zip" --limit 20
```

最后一行是路径示例，请替换为自己机器上的 ZIP 或目录。不要根据 Mac 上的测试结果假定 Windows 的 PyTorch/GPU 已正常。

## 4. 可选：保存每台机器自己的路径

将根目录 `config.example.json` 复制为 `config.local.json`，填写：

```json
{
  "data_path": "D:/datasets/Stanford_Online_Products.zip"
}
```

JSON 中 Windows 路径建议用 `/`，或将反斜杠写成 `\\`。`config.local.json` 已被 Git 忽略，不影响其他人的路径。也可以设置 `SOP_PATH` 环境变量，它优先于本地配置。

配置后可运行：

```bash
python -m sop_project.data
python -m sop_project.data --full
```

`--full` 会解码所有原图，可能需要较长时间。报告会区分 full_dataset 与 prefix_smoke_test_only，列出不足尺寸和无法读取的记录。不会放大、修改或删除原图。

## 5. 运行后的交接

- 组内确认环境报告、数据图片/商品总数、是否成功显示图片。
- 然后再确认 PyTorch 安装版本、公共特征提取与正式商品级划分。
- 共享代码以 `PythonCode/` 的 `.py` 为准；若改动 Notebook，提交前清除输出，尤其是显示的原图；数据、配置和缓存仍保存在本机。
- `processed_sop/`、ZIP 和缓存由 `.gitignore` 排除；提交前仍检查 Git 状态，确认没有数据或生成结果进入提交。
- 运行 `python -m unittest discover -s tests -v` 可检查数据工具。测试只临时生成合成图片，不需要真实数据集。

## 验证记录

2026-10-11：Mac 上 8 项数据工具测试通过；真实 SOP 原包索引读取 120053 张图片、22634 个商品，前 20 张解码与尺寸检查通过。覆盖 ZIP、目录及其父目录、坏图和缺失图片、重复 ID、商品跨原始划分、危险路径、本地配置与环境变量。

Notebook 格式校验通过，全部代码格已在 Mac 的 Python 3.12.14 / Pillow 12.3.0 中使用真实 SOP 依次执行，默认安装开关关闭；图片对象成功读取为 400x268。未保存数据图片或执行输出到仓库。环境诊断也在另一个 Python 3.9.6 解释器中运行通过，能报告缺失依赖及 CPU-only PyTorch。

这些不是 Windows 实测，也不是完整模型训练。依赖范围尚未锁定；主运行机验证后应另存可复现环境版本。[环境指南](environment.md)提供官方后端选择流程与命令，但Windows/CUDA仍待实测，不能当作已验证记录。
