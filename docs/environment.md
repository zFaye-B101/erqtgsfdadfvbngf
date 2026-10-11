# 环境配置与验收

## 最新依赖入口（v3）

在最终运行机上，先用将要运行实验的同一解释器执行 `python PythonCode/00_Check_Environment.py`（或 PyCharm 的 `00 Check environment` 配置）。它只读检查依赖、PyTorch/CUDA 构建、数据位置、磁盘空间和 Windows 路径长度，不安装、不下载。Windows PyCharm 预置见 [quickstart](quickstart.md)。

PythonCode正式流程使用Python3.11+，Windows CPU合成检查的实际版本见[VALIDATION](../PythonCode/VALIDATION.txt)。基础依赖以PythonCode/requirements-base.txt为唯一共同清单，包含LR/SVM和绘图；根requirements-base.txt通过-r引用该清单并追加历史Notebook内核，requirements-experiments.txt再引用根清单。torch/torchvision仍按设备单独安装，既有OpenCV不要重复叠装。下方历史开发环境说明不替代v3的验证记录，真实数据和GPU仍待验收。

适用：当前SOP读取、HSV/HOG/SIFT、Notebook及冻结ResNet18开发诊断。特征和训练实现仍在讨论；这不是SupCon训练验收，也不是最终提交的英文Readme.txt。正式运行入口为 `PythonCode/` 的 `.py` 脚本，只需本环境的解释器；下文Notebook内核一节仅在使用历史/预览Notebook时需要。

## 1. 依赖与边界

建议统一使用 **64位Python3.12**。Git同步代码；VS Code/PyCharm只是编辑器，不代替Python环境。每人自己的环境、数据路径和硬件后端分开配置，不共享`.venv`，不上传数据、权重和缓存。终端命令在Terminal运行，不是Python控制台。

| 依赖 | 用途 | 安装入口 |
| --- | --- | --- |
| Python3.12、pip、venv | 解释器、安装与环境隔离 | [Python官网](https://www.python.org/downloads/)或已有Homebrew Python |
| NumPy、pandas、Pillow | 数组、预处理表格、读图 | `requirements-base.txt` |
| opencv-python-headless | HSV、梯度、SIFT，导入名`cv2` | `requirements-base.txt` |
| ipykernel | Notebook内核 | `requirements-base.txt` |
| scikit-learn、Matplotlib | 开发诊断、候选基线工具、绘图 | `requirements-experiments.txt`，包含基础依赖 |
| torch、torchvision | CNN和学习型模型基础 | 按硬件单独安装，见第3节 |
| JupyterLab | 浏览器Notebook界面 | 可选；IDE可运行Notebook时不必安装 |

`csv/json/pathlib/zipfile/random/collections/time`属于标准库，不需要pip。SciPy等传递依赖由pip安装。当前不需要TabM、TensorFlow、torchaudio、DINOv2、LightGlue或额外的“SupCon包”；安装torch不代表已经实现SupCon。

**课程限制：** 安装scikit-learn不代表所有接口可用于正式提交。此前`roc_auc_score`只用于开发诊断；正式交叉验证、调参、嵌套CV及评价指标需自行实现。见[官方要求](../materials/official/Project.md)及[要求核对](reviews/2026-10-11_9291720_course-requirements_codex.md)。

## 2. 创建独立环境

先进入仓库根目录，确认能看到`requirements-experiments.txt`。已有克隆先保留本地修改，不能为拉取强制覆盖Notebook。以下创建`sml-project`；若同名环境已有内容，先检查或统一换名，不要覆盖现有环境。

### macOS / Linux

先确认`python3.12 --version`。没有时先安装Python3.12；Apple Silicon已有Homebrew Python时可将首行解释器换成`/opt/homebrew/bin/python3.12`。

```bash
python3.12 -m venv "$HOME/.venvs/sml-project"
source "$HOME/.venvs/sml-project/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-experiments.txt
```

### Windows PowerShell

安装64位Python3.12和Git后重新打开终端，先确认`py -3.12 --version`。下面直接调用环境内Python，不需要修改PowerShell执行策略。

```powershell
py -3.12 -m venv "$env:USERPROFILE\.venvs\sml-project"
$py = "$env:USERPROFILE\.venvs\sml-project\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements-experiments.txt
```

后续`$py`都指这个环境；新终端需要重新赋值。不要把Mac的`source`或`python - <<'PY'`原样贴到PowerShell。

## 3. PyTorch后端：三选一

基础依赖不包含torch，避免CUDA机器误装CPU版。清单是开发范围，不是最终锁文件；实际安装版本须记录。安装方式依据[官方安装页](https://pytorch.org/get-started/locally/)及[指定版本命令](https://pytorch.org/get-started/previous-versions/)。

### A. Mac（Apple Silicon用MPS，不安装CUDA）

```bash
python -m pip install torch torchvision
```

### B. Windows / Linux CPU

Windows：

```powershell
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Linux已激活环境：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### C. Windows NVIDIA GPU

先运行`nvidia-smi`确认显卡和驱动，再在[官方选择器](https://pytorch.org/get-started/locally/)选择Windows/Pip/Python及驱动兼容的CUDA构建。不猜CUDA后缀、不照抄Mac版本号。可将官方安装命令的`pip3`改为`& $py -m pip`；本项目不需要torchaudio。

也可将官方命令中的`--index-url`地址输入以下代码。代码仅校验官方CUDA索引格式，不自动判断驱动兼容性或安装驱动。

```powershell
$TorchIndex = (Read-Host "Paste the CUDA index URL from the official PyTorch selector").Trim()
if ($TorchIndex -notmatch '^https://download\.pytorch\.org/whl/cu[0-9]+/?$') { throw "Use an official CUDA wheel index URL." }
& $py -m pip install torch torchvision --index-url $TorchIndex
```

torch/torchvision交给同一次pip安装解析兼容版本。GPU安装失败保留错误，不随意混搭版本或把CPU回退当成GPU配置成功。Windows/CUDA本项目尚未实测。

## 4. Notebook内核

Mac/Linux：

```bash
python -m ipykernel install --user --name sml-project --display-name "Python (SML Project)"
```

Windows：

```powershell
& $py -m ipykernel install --user --name sml-project --display-name "Python (SML Project)"
```

VS Code/PyCharm选择 **Python (SML Project)** 或环境内解释器。注册内核不等于已切换；在Notebook运行`import sys; print(sys.executable)`确认路径指向`sml-project`，不能仍是系统Python3.9或旧`.venv`。装包后按需重启内核。

没有可用Notebook界面时，可选`python -m pip install jupyterlab`和`python -m jupyterlab`；Windows改为`& $py -m pip install jupyterlab`、`& $py -m jupyterlab`。

## 5. 可直接运行的验收格

在选好的Notebook内核执行。不下载模型、不读SOP；除导入外，实际执行SIFT创建及小矩阵运算。GPU操作失败保留完整错误，不静默回退。

```python
import sys
import numpy as np
import pandas as pd
import PIL
import cv2
import sklearn
import matplotlib
import ipykernel
import torch
import torchvision

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
for name, module in [("numpy", np), ("pandas", pd), ("Pillow", PIL),
                     ("OpenCV", cv2), ("scikit-learn", sklearn),
                     ("matplotlib", matplotlib), ("ipykernel", ipykernel),
                     ("torch", torch), ("torchvision", torchvision)]:
    print(name, module.__version__)
assert cv2.SIFT_create() is not None
print("SIFT available:", True)
print("CUDA available:", torch.cuda.is_available())
print("CUDA build:", torch.version.cuda)
print("MPS available:", torch.backends.mps.is_available())
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.ones((4, 4), device=device)
y = (x @ x).cpu()
assert torch.allclose(y, torch.full((4, 4), 4.0))
print("Tensor test passed on:", device)
```

终端再执行`python -m pip check`和`python -m sop_project.environment`；Windows改为`& $py -m pip check`、`& $py -m sop_project.environment`。现有环境模块未检查全部依赖/MPS，不能代替上面格子。小矩阵通过不等于CNN前向、训练、全量Notebook或正式实验通过。

## 6. 数据路径与只读测试

SOP ZIP不上传、不必解压。从仓库根目录执行，替换为本机实际路径：

```bash
python -m sop_project.data --data "/path/to/Stanford_Online_Products.zip" --limit 20
python -m unittest discover -s tests -v
```

Windows：

```powershell
& $py -m sop_project.data --data "D:/datasets/Stanford_Online_Products.zip" --limit 20
& $py -m unittest discover -s tests -v
```

持久路径见[quickstart](quickstart.md)。前20张读取通过不等于全量核验；不要为了验收环境直接跑全量预处理Notebook，它会导出图片和特征并消耗磁盘。

## 7. 版本记录与交接

验收后在仓库根目录导出记录，`outputs/`已被忽略：

```bash
mkdir -p outputs
python -m pip freeze > outputs/requirements-local.txt
python -m sop_project.environment > outputs/environment-local.json
```

Windows：

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
& $py -m pip freeze | Out-File -Encoding utf8 outputs/requirements-local.txt
& $py -m sop_project.environment | Out-File -Encoding utf8 outputs/environment-local.json
```

交接包版本、硬件、验收格及数据测试结果。记录可能含本机路径/私有索引，审查后再分享。`pip freeze`不记录驱动/安装索引，不是跨平台通用安装方案；最终主运行机完整验证后再提交清理后的锁文件和英文运行说明。

## 8. 已有环境与常见问题

- **已有`sml-cnn`能跑：** 不必覆盖它。当前缺pandas、matplotlib、ipykernel，可单独执行`python -m pip install pandas matplotlib ipykernel`，再注册对应内核。这不是升级torch的命令。
- **OpenCV：** GUI/headless及contrib变体共享`cv2`，每个环境只保留一种。新环境使用headless即可；已有GUI版时不要直接叠装基础清单。迁移前先记录版本，再在该虚拟环境卸载冲突变体并重装一种，不动系统Python。
- **装包后仍找不到：** 核对终端/Notebook的`sys.executable`，不要用裸`pip`给其他解释器安装。
- **SIFT不可用：** 检查实际cv2版本及冲突包，不能以全零向量掩盖缺失实现。
- **模型权重：** 首次ResNet18会下载约45MB到用户缓存，需要联网。下载失败不能改用随机权重冒充预训练，不上传缓存。
- **Windows GPU不可用：** 保留`nvidia-smi`、torch版本及CUDA build输出，检查驱动和构建。CPU可做小测试，不等于GPU已验收。

## 9. 当前验证边界（2026-10-11）

- 用户Mac arm64：Python3.12.14、torch2.14.1、torchvision0.29.1，MPS前向及小样本CNN/传统特征诊断通过。这是已报告的具体环境，不保证这些版本在所有平台可用。
- 本轮只读包元数据：numpy2.5.4、Pillow12.3.0、opencv-python5.0.0.93、scikit-learn1.9.1；该`sml-cnn`尚无pandas、matplotlib、ipykernel。其他环境装过不代表这里已安装。
- 上述实测使用GUI版OpenCV5；新环境沿用基础清单的headless4.x，是不同配置，不能互相替代验收。依赖安装范围不是最终版本锁。
- 新环境完整安装、Windows CPU/CUDA、全量Notebook、SupCon训练及正式复现实验仍待执行。终端试验代码在会话中，不因发布本指南就成为仓库里的训练脚本。
