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

`notebooks/01_SOP_Preprocessing.ipynb` 直接读取 ZIP，无需先解压。它检查索引图片的存在性、可读性和原始尺寸，删除不合格图片及清理后仅剩一张图片的商品，再保留官方 train/test 标记并导出 224×224 RGB 图片。它另计算 48 维 HSV 颜色直方图和 144 维空间 HOG 特征；`notebooks/sop_features.py` 还提供 SIFT 图片对特征函数供后续实验调用。这些是现有代码产物，不代表最终实验特征比较方案已经锁定。

运行时会把图片、CSV 审计/清单和 `.npz` 特征保存到当前工作目录下的 `processed_sop/`。该目录及 ZIP/缓存由 `.gitignore` 排除。不要提交原图、数据 ZIP、完整派生特征缓存或 Notebook 内嵌样例图片；文本统计和核验结果可以共享。

如作者链接失效，应回作者入口核对，不把未经核验的第三方镜像当作同一个数据包。
