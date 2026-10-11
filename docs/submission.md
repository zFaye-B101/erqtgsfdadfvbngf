# 最终代码包与 Notebook 数量

核对日期：2026-10-11。依据仓库已归档的 [Project.pdf 第1页](../materials/official/Project.pdf)、[全文提取版](../materials/official/Project.md) 和 [Canvas 提交页快照](../materials/official/canvas-page-2026-09-27.md)。本次不是实时 Canvas/Ed 巡检；若课程发布新规定，应以新规定为准。

## 官方原文要求

> A ZIP archive of your Python code or Jupyter notebook and a Readme.txt file (describing in just a few lines how to run the code) and any scripts for automation.

同时提交四页 PDF 报告；不要提交数据。原文没有规定“只允许一个 .ipynb”，没有给 Notebook 文件数上限，也没有要求将所有代码放进一个文件。它明确允许 Python 代码及自动化脚本。

因此本组按多文件代码包组织，不因单数用词 notebook 就额外推导出“只能一个”的限制。这里没有声称 staff 单独回复批准了某种文件数量。

## 本组安排

- 开发阶段允许多个 `.py` 和 Notebook；不为满足不存在的单文件要求而手工拼接 JSON。
- 建议最终保留一个清晰的运行入口，调用公共模块；确需多个 Notebook 时在 `Readme.txt` 标明先后顺序、输入与输出。单入口是便利复现的建议，不是官方硬性要求。
- 正式 ZIP 只装运行所需代码、必要依赖/配置示例、`Readme.txt`、自动化脚本和按官方要求允许的协作记录。不要直接把整个资料仓库打包提交。
- 保留相对路径和本机数据路径配置说明；不要求老师拥有组员电脑的绝对路径或私有账号。
- 在新会话/干净内核按说明验证，不能依赖某位组员之前运行留下的变量。
- 开发中各人可以运行自己的 Notebook；最终结果必须共用已确认的数据划分、协议和评价流程，而不是三个独立实验各交一份。

## Git 协作决定

用户选择直接更新 `main`，不要求为同一 Notebook 开分支再 merge。Git 分支策略与最终文件数量没有必然关系。按文件分工、避免并发编辑同一个 Notebook，提交前同步 README 与相关文件，推送前拉取合并队友的新提交；发生冲突时保留双方内容，不强制覆盖。

此说明只确认交付组织方式；不代表模型代码、完整运行入口或最终提交包已经完成。
