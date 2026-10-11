# 协作与运行路径

## 同步范围

- 所有获得仓库访问权限的组员可下载这里的代码、Markdown 和原始资料，不需要原作者的本地路径。
- Canvas 链接仅供来源追溯，可能需要学校登录；已归档原件不依赖 Canvas 登录。
- SOP 数据另从作者渠道下载。每台机器通过 Notebook 提示输入自己的 ZIP 路径，不把本机绝对路径写入共享代码。
- 不共享 GitHub token、密码、`.env`、私钥或个人服务配置。

## 多人修改

1. 开始前先同步最新 `main`。
2. 每人使用自己的工作分支，避免同时改同一个 Notebook。
3. 按数据/特征、基线/评估、SupCon 分工；具体文件接口需共同确认后再实现。
4. 提交前清除 Notebook 图片输出，检查是否误加入原图、压缩包、缓存或模型权重。
5. 合并时注明改动及运行验证状态；仅写好代码不能记为测试通过。

当前基础环境入口见 [quickstart.md](quickstart.md)，`requirements-base.txt` 仅包含图片读取和 Notebook 依赖；模型训练依赖版本仍待验证。`tools/requirements-docs.txt` 仅供文档转换。

## 现有 Notebook

新的共享入口是 `notebooks/00_setup_and_data.ipynb`，支持 ZIP 和解压目录，按单元格从上往下运行。`materials/sop_data_check.ipynb` 保留为早期数据核验记录。Windows 依赖仍待验证；运行后显示的本机原图不能随 Notebook 输出提交到 GitHub。

仓库仍为私有；添加新成员或公开仓库需要单独确认。不要将课程模板或第三方资料再公开分发。
