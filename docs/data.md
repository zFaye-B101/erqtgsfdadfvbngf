# SOP 数据下载与资格

## 最新 v3 修复（2026-10-11，覆盖此前 v2 的对应状态）

- 新增 handcrafted_content 和 handcrafted_padding_audit，支持排除填边的HSV/HOG以及与原填边版本的配对对照；原handcrafted保持为对照，ResNet仍使用明确记录的填边输入。真实SOP收益未量化。
- 每个外层训练商品集合再独立预留20%校准商品（每大类至少2个，保留足够拟合商品）。校准商品不参与内层选参、归一化或训练；最终模型在其上定阈值后不再重训，直接评估外层测试。有限样本校准不保证测试召回率达到95%。
- SupCon温度边界胜出时按内层验证扩展三点窗口，最多两次、因子2、温度限制0.005至1.0；复用已评估候选、保存搜索历史并保留未解决的边界警告。
- 官方holdout默认关闭。新协议/缓存/结果使用v3目录，不能复用v2阈值或断点；已有01处理图片可继续使用。入口整理为01预处理和02统一实验命令（--method baselines|supcon），选特征和参数仍改配置。
- Windows CPU合成检查覆盖填边探针、阈值校准隔离、温度扩展和端到端流程，验证边界见[VALIDATION](../PythonCode/VALIDATION.txt)。真实SOP十折、正式holdout、预训练ResNet及GPU仍未验证。下方此前状态保留为历史记录。


## 当前 PythonCode 数据接口（2026-10-11）

新01仅完成原图资格检查、单图商品排除、224x224填边RGB JPEG和审计清单导出。统一实验入口02通过data_dir读取清单及图片，按配置重新构建共同干净/扰动特征缓存；原NPZ不再读取，因此已有合格处理目录可复用而无需重跑01。所有条件从同一JPEG解码出发，白色填边仍进入特征，不能声称其影响已解决。缓存、共同配对协议和模型输出留在本机。下文Notebook的NPZ说明仅适用于历史入口。


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
