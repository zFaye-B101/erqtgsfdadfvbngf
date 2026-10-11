可先运行 00_Check_Environment.py 做只读环境检查。共享 .run 配置默认选择 F1，可修改 --group。

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
python 02_SOP_Experiments.py --group all --method supcon

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
