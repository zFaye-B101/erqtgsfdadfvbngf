# SOP 数据下载与资格

数据集不在仓库内，也不应放入最终提交包。

- [作者的数据集入口](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16#stanford-online-products-dataset)
- [作者提供的下载链接](https://drive.google.com/uc?export=download&id=1TclrpQOF_ullUP99wk_gjGN8pKvtErG8)
- [数据资格核验记录](../materials/research/stanford-online-products-audit.md)

已核验原包：3,083,860,082 bytes。

```text
SHA-256 b04fa78210e69fab050ce73939550e1cd0f96890bdd98ec9fee732c3b756ef48
```

原包包含 120053 张图片、22634 个商品。120026 张原图宽高均达到 100；27 张不足，应排除而不是放大来达标。筛后 22630 个商品有至少两张图，4 个只剩一张，不能自行构成正例。

现有 Notebook 直接读取 ZIP，无需解压。保存位置由各机器自行决定，运行时输入完整 ZIP 路径。不要提交原图、数据 ZIP、完整派生特征缓存或 Notebook 内嵌样例图片；文本统计和核验结果可以共享。

如作者链接失效，应回作者入口核对，不把未经核验的第三方镜像当作同一个数据包。
