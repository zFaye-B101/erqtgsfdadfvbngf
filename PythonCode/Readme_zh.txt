白边配置拆分（2026-10-14）
运行电脑只替换02，新增F1.1和F2.1两份JSON，删除旧F5 JSON；01、03、config.json和utils不替换。
运行：python 02_SOP_Experiments.py --group F1.1,F2.1 --method all
F1.1和F2.1是正式组名；默认结果进入results/Main_Experiments。
先把旧results/Main_Experiments/F5_Traditional_Content_HSV_Gradient移到results之外备份。
新两组完成后，先将原results_recall90改名，再运行：
python 03_SOP_Recalibrate_90.py --source results --output results_recall90
03读取所有完整结果重新算90%指标，不重训；已有输出目录会拒绝覆盖。
八组指F1/F1.1/F2/F2.1/F3/F4/F4.1/F4.2；不包含未运行的F6。不同运行名下重复的F3/F4仍保留。
以下F5说明是历史记录，已有实验归档不会被本次配置修改改写。

90%目标召回率的补充分析（不用重新训练）
原来的95%代码、配置、结果都保留。运行：
python 03_SOP_Recalibrate_90.py
默认读取同目录 results.zip；没有ZIP时读取 results 文件夹。
只写入新 results_recall90 文件夹。若新目录已经存在，会停止，不覆盖。
原结果不用删除、移动或改名。以后重复分析可用 --output 指定新目录。
输出含全部八项指标、图表、新预测、校准记录、95%与90%逐折对照。
90%是干净校准集召回目标，不保证扰动测试召回90%或FPR低于50%。
这是已有模型的阈值敏感性分析，不是新训练轮次。

可先运行 00_Check_Environment.py 做只读环境检查；默认检查全部组，也可用 --group F1,F2（与 02 相同写法）只检查计划要跑的组。共享 .run 配置默认选择 F1，可修改 --group。

只需运行 01 和 02

新增局部方案A/B（2026-10-12）
已确认F7与F4.1计算相同，删除F7配置，只保留F4.1；原F8改名F4.2。
F6保留组员的ResNet50全局2048；A使用F4.1，B使用F4.2，不覆盖已有编号。
F4.1：F4的全局512＋四个区域平均描述子的逐通道最大值512，共1024维。
F4.2：保留同一全局512；四角和中心五个160×160窗口分别缩放、经同一ResNet18
编码，再逐通道最大汇总为局部512，共1024维。按窗口分批前向，不一次堆六倍批量。
四类扰动仍在查询图上先施加，再取窗口；不使用遮挡掩码挑选可见区域。
F4.1复用当前代码版本且身份校验通过的resnet18缓存；旧版本缓存不强行复用。
F4.2需要重新提取局部裁剪特征。01处理图片无需重跑。
不要把F4.1/F4.2加进Main_Experiments：该运行名是已结束的第1轮（R1），而且自动跨组比较会跳过新旧代码指纹不同的组。第2轮（R2）用一个新运行名把基准和两个局部方案一起重跑（规则见docs/versioning.md）：
python 02_SOP_Experiments.py --group F3,F4,F4.1,F4.2 --method all --run-name R2_LocalFeatureAB
（Local_Pilot是此前的示例名，不是已登记的轮次。）
新增提取器改变代码/缓存指纹，不要用新版恢复同一组的旧版结果，不要绕过检查。
这次除了新增两个JSON，还实现提取器、注册与组件变换和预检权重信息；02入口接口不变。
真实GPU权重运行和正式效果尚未验证，不能保证加入局部特征一定改善。

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

第 2 轮（R2）：局部特征方案 A（F4.1）与 B（F4.2），连同基准 F3/F4。F4.1 = F3 全局 512 维 + F4 四个区域向量逐通道最大值；F4.2 = 同一全局 512 维 + 五个窗口分别编码后取最大值；均为 1024 维。代码指纹已变化，F3/F4 在 R2 中重跑：
python 02_SOP_Experiments.py --group F4.1,F4.2 --method all --smoke-test --skip-official-test
python 02_SOP_Experiments.py --group F3,F4,F4.1,F4.2 --method all --run-name R2_LocalFeatureAB
命名约定：新方法用新编号（F7、F8……）；对已有方法的改进用子版本（F4.1、F4.2……）。正式运行按轮次编号 R1、R2……，运行名写成 R<轮次>_<主题>（不含点）；第一轮 F1–F4 即已归档的 Main_Experiments。
运行结束：所有组和跨组比较保存完毕后，02 会打印 “All requested runs finished. Results: <运行目录>”，此后可以放心停止进程。同时在运行目录写入 exit_watchdog_traceback.txt：正常退出时自动删除；若 120 秒后进程仍未退出，所有线程的调用栈会写入该文件，便于排查卡住原因，结果不受影响。

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
