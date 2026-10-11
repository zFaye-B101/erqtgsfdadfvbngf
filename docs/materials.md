# 资料索引与转换边界

## 官方材料

| 内容 | 可搜索版本 | 原件 |
| --- | --- | --- |
| 作业说明，4 页 | [Project.md](../materials/official/Project.md) | [Project.pdf](../materials/official/Project.pdf) |
| 小组协议空白模板 | [GroupAgreement.md](../materials/official/GroupAgreement.md) | [GroupAgreement.docx](../materials/official/GroupAgreement.docx) |
| 研究问题空白模板 | [Questions.md](../materials/official/Questions.md) | [Questions.docx](../materials/official/Questions.docx) |
| 会议记录空白模板 | [MeetingMinutes.md](../materials/official/MeetingMinutes.md) | [MeetingMinutes.docx](../materials/official/MeetingMinutes.docx) |

[要求摘要](../materials/official/REQUIREMENTS.md)及[Canvas 页面快照](../materials/official/canvas-page-2026-09-27.md)也已保留。快照日期不是实时巡检日期，旧提交状态不能作为当前状态。

## 论文

| 论文 | 用途 | Markdown | PDF |
| --- | --- | --- | --- |
| Supervised Contrastive Learning, NeurIPS 2020 | 当前选定方法 | [SupCon](../papers/SupCon-2020.md) | [原件](../papers/SupCon-2020.pdf) |

SupCon 原件下载自 [NeurIPS 官方 PDF](https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf)，实现参考 [论文所列 PyTorch 实现](https://github.com/HobbitLong/SupContrast)。按用户最新要求，不上传或保留 ProtoNet、ProxyGML 的提取稿及 PDF。SOP [数据集论文](https://cvgl.stanford.edu/papers/song_cvpr16.pdf)先保留引用入口。论文供组内阅读与引用，未将他人实现冒充本组代码。

## 代码与历史材料

- [数据核验 Notebook](../materials/sop_data_check.ipynb) / [代码导出](../materials/sop_data_check.md)。上传副本清除了输出、附件及单元格运行元数据，源代码没有改动。
- [SOP 资格核验](../materials/research/stanford-online-products-audit.md)。
- [早期 SOP 提案](../materials/planning/2026-09-25-sop-startup.md)、[早期启动计划](../materials/planning/2026-09-25-startup.md)、[UCI HAR/SCARF 备选](../materials/research/uci-har-scarf.md)均为历史记录。
- 用户于 2026-10-11 明确确认删除了 `墨大COMP90051A2P1.pptx`，随后在提交 `7cc8e2b` 中清空了 `materials/guidance/README.md`。保留该修改，不恢复 PPT 或说明内容。
- [旧目录首页](../materials/README.md)已标注为历史版本。

## 转换和完整性

- PDF 按原页顺序提取全文，保留页码标题和等宽文本；公式排版、图表和扫描文字不保证完整还原。涉及公式、图形或排版时查看原件，机器提取稿不是数学转录。
- DOCX 提取正文和表格；空白模板的手写空间及页码域不等于缺失内容。原件保留，可继续填写。
- 原件未改写。Markdown 是阅读副本，不是官方新版本。
- [机器清单](materials-manifest.json)列出来源、上传路径、哈希及排除原因。
- 修改归档材料时，按 [协作规则](../AGENTS.md) 同步本索引与机器清单；保留原始哈希并核验当前副本哈希，不自动从旧本地目录恢复已删除内容。
- 本轮范围是 A2 作业目录和当前选定的 SupCon 论文，不是整门课程或其他作业。
- 排除 SOP 数据集、系统 `.DS_Store`、零字节误建文件、Notebook 嵌入图片/输出，以及用户已删除的 PPT。没有删除本地源文件。
