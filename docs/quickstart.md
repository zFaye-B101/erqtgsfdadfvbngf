# 从 GitHub 拉取并运行

本阶段仅检查环境和数据。三模型、正式划分、特征提取和嵌套评估尚未实现，不要把通过数据读取当成实验完成。

## 1. 获取仓库

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

本组当前直接更新 `main`，不要求个人分支或 PR。各人负责不同文件，避免同时编辑一个 Notebook。自己提交完成且工作区干净后，先 `git pull --rebase origin main` 再 `git push origin main`；发生冲突先处理，不强制推送。详见 [协作规范](../CONTRIBUTING.md)。

## 2. 在 PyCharm 打开整个仓库目录

选择本项目的 Python 解释器，并打开 [00_setup_and_data.ipynb](../notebooks/00_setup_and_data.ipynb)。从第一个代码格开始运行。

- 第一格输出当前内核、Python、依赖导入结果与 GPU 信息，不自动安装软件。
- 缺少基础依赖时才启用安装格的 `INSTALL_BASE = True`，安装到当前内核。基础包不包含 PyTorch。
- 后续格会提示输入 SOP ZIP 或已解压文件夹的完整路径。支持直接包含 `Ebay_train.txt` / `Ebay_test.txt` 的目录，或它上一级且子文件夹叫 `Stanford_Online_Products` 的目录。
- 检查前 20 张图片并显示第一张。前 20 张只是读取测试，不是全量资格核验，也不能用于正式划分。

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
- 提交 Notebook 前清除输出，尤其是显示的原图；数据、配置和缓存仍保存在本机。
- 运行 `python -m unittest discover -s tests -v` 可检查数据工具。测试只临时生成合成图片，不需要真实数据集。

## 验证记录

2026-10-11：Mac 上 8 项数据工具测试通过；真实 SOP 原包索引读取 120053 张图片、22634 个商品，前 20 张解码与尺寸检查通过。覆盖 ZIP、目录及其父目录、坏图和缺失图片、重复 ID、商品跨原始划分、危险路径、本地配置与环境变量。

Notebook 格式校验通过，全部代码格已在 Mac 的 Python 3.12.14 / Pillow 12.3.0 中使用真实 SOP 依次执行，默认安装开关关闭；图片对象成功读取为 400x268。未保存数据图片或执行输出到仓库。环境诊断也在另一个 Python 3.9.6 解释器中运行通过，能报告缺失依赖及 CPU-only PyTorch。

这些不是 Windows 实测，也不是完整模型训练。基础依赖范围尚未锁定；主运行机验证后应另存可复现环境版本，不提前生成未经验证的 GPU 安装方案。
