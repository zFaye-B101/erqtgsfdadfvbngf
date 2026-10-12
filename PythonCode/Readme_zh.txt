可先运行 00_Check_Environment.py 做只读环境检查；默认检查全部组，也可用 --group F1,F2（与 02 相同写法）只检查计划要跑的组。共享 .run 配置默认选择 F1，可修改 --group。

只需运行 01 和 02

1. PyCharm 解释器选择 Unimelb_SML_A2。
2. 01 已经成功处理数据，就跳过。本次只修改工具导入路径，不改变处理逻辑。
3. 右键运行 02_SOP_Experiments.py，在控制台选择组、模型和运行名称。
   组可以输入 1、1,2、F1,F2 或 all。
   模型：1 为逻辑回归和 SVM；2 为 SupCon；3 为全部。
   运行名称直接回车使用 Main_Experiments。

Terminal 示例：
python 02_SOP_Experiments.py --group F1,F2 --method baselines
python 02_SOP_Experiments.py --group F3,F4 --method baselines
python 02_SOP_Experiments.py --group F2,F5 --method baselines   （白边对照：F5 只用商品内容区域）

局部特征方案 A（F4.1，当前优先）：F4.1 = F3 的全局 512 维 + 对 F4 四个区域向量逐通道取最大值（512 维），共 1024 维。这一版代码改变了模型源码指纹，所以用同一个运行名把 F3、F4 和 F4.1 一起重跑：
python 02_SOP_Experiments.py --group F4.1 --method all --smoke-test --skip-official-test
python 02_SOP_Experiments.py --group F3,F4,F4.1 --method all --run-name R2_LocalFeatureA
命名约定：新方法用新编号（F7、F8……）；对已有方法的改进用子版本（F4.1、F4.2……）。正式运行按轮次编号 R1、R2……，运行名写成 R<轮次>_<主题>（不含点）；第一轮 F1–F4 即已归档的 Main_Experiments。

ResNet50 小试（F6 对 F3，保留但不是当前优先）：F6 使用冻结的 ResNet50 IMAGENET1K_V2 权重（首次使用时 torchvision 自动下载约 98 MB），输入与 F3 相同（带白边的 224 图）。新增 ResNet50 提取器改变了模型源码指纹，新结果不能和 2026-10-12 的 F1–F4 结果直接配对，因此用新的运行名把 F3 和 F6 一起重跑（共同协议和测试图片对不变）：
python 02_SOP_Experiments.py --group F6 --method all --smoke-test --skip-official-test
python 02_SOP_Experiments.py --group F3,F6 --method all --run-name R3_ResNet50Pilot
python 02_SOP_Experiments.py --group all --method supcon

启动 SupCon/ResNet 实验时，02 会先加载 PyTorch；含 ResNet 的实验还会检查 torchvision。
启动检查会执行一次小型张量运算，打印实际 Python 路径和设备（cuda/cpu 等）。
检查失败会在任何实验组开始之前停止，避免 baseline 完成后才首次加载 PyTorch。
仅传统特征 baseline、--list 和 --compare-only 不要求这项检查。
这次只修正入口的启动流程，不改变模型及协议的代码指纹；配置、数据和记录的
软件版本一致时，已完成的折仍可复用。

代码可传到另一台 GPU 电脑运行，本开发电脑只负责整理和分析回传结果。
F1–F4 三算法的完整十折结果位于仓库根目录 docs/experiments/2026-10-12-main-experiments。
公开副本保留指标、预测、图表及调参记录，机器路径改为相对路径，不包含模型。
它用于查看和核对结果，不能代替 GPU 电脑上包含模型的原始续跑目录。

utils 都是内部工具，01 直接从 utils 导入。根目录兼容文件和旧入口备份已删除。
tests 是开发检查，不需要放入日常实验运行顺序。
公共设置在 config.json，各组特征在 experiment_configs/*.json。
新增有效配置后，02 自动显示。全新算法还需实现并注册提取器。
已有特征组合只需新增配置，格式见 experiment_configs/README.md。

结果在 results/运行名称/实验组/baselines 或 supcon。
protocol.json 保存完整实际配置、代码指纹和软件版本。
comparisons 自动汇总已完成的兼容实验，生成逐折差异，不重复训练。
相同设置可恢复已完成折；完整重跑用新名称，如 --run-name HSV48_repeat1。
相同随机种子不构成独立统计重复。复原实验还需保留代码与环境版本。

已有图片和旧结果保持不动。本次移动工具改变代码指纹，因此使用新协议目录。
旧、新代码版本结果可能不能自动合并。改划分、种子或扰动参数时，所有对比组
必须使用相同的新协议目录。默认 10 折外层、3 折内层，官方 test 最终测试关闭。
