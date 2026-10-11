# 协作与运行路径

每次开始前读 [AGENTS.md](../AGENTS.md) 和 [CONTRIBUTING.md](../CONTRIBUTING.md)。每次更新共享文件，必须在同一提交/PR内同步根目录 README 及受影响的相关文件；交接时说明验证结果和未完成项。

## 同步范围

- 当前仓库地址为 [zFaye-B101/erqtgsfdadfvbngf](https://github.com/zFaye-B101/erqtgsfdadfvbngf)，2026-10-11 核验为 Public，任何人都可下载；推送仍需要相应写入权限。
- Canvas 链接仅供来源追溯，可能需要学校登录；已归档原件不依赖 Canvas 登录。
- SOP 数据另从作者渠道下载。每台机器通过 Notebook 提示输入自己的 ZIP 路径，不把本机绝对路径写入共享代码。
- 不共享 GitHub token、密码、`.env`、私钥或个人服务配置。

## 多人修改

1. 默认所有组员直接更新 `main`；工作区干净时开始前先 `git pull --ff-only origin main`。
2. 不要求每人开分支或 PR，按文件分工，避免同时改同一个 Notebook。
3. 按数据/特征、基线/评估、SupCon 分工；具体文件接口需共同确认后再实现。
4. 提交前清除 Notebook 图片输出，检查是否误加入原图、压缩包、缓存或模型权重。
5. 代码与 README/相关文件放在同一次提交。提交后工作区干净时 `git pull --rebase origin main`，解决冲突再 `git push origin main`；不 force push，不覆盖队友改动。
6. 交接时注明改动及运行验证状态；仅写好代码不能记为测试通过。提交形式详见 [submission.md](submission.md)，不必强行合成一个 Notebook。

当前基础环境入口见 [quickstart.md](quickstart.md)，`requirements-base.txt` 仅包含图片读取和 Notebook 依赖；模型训练依赖版本仍待验证。`tools/requirements-docs.txt` 仅供文档转换。

## 现有 Notebook

`notebooks/00_setup_and_data.ipynb` 用于环境检查和前 20 张图片预览，可选运行。`notebooks/01_SOP_Preprocessing.ipynb` 才执行全量 ZIP 图片审计、筛选、224×224 导出和 HSV/HOG 特征缓存；需与 `notebooks/sop_features.py` 一起使用。输出位于本机 `processed_sop/`，已由 `.gitignore` 排除。`materials/sop_data_check.ipynb` 保留为早期核验记录。Windows 全量预处理仍待运行验证；目前没有模型训练 Notebook。

仓库原为私有，现由用户改为 Public。未来任何权限或可见性变更仍需单独确认，不由 AI 擅自调整；现有资料公开不等于获得再分发授权。
