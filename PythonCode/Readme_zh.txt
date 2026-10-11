SOP 商品匹配实验流程

一、运行入口
01_SOP_Preprocessing.py：检查原始 ZIP 图片，排除无法读取或缺失的图片、任一边小于 100 像素的图片，以及合格图片少于两张的商品。导出带白色填边的 224×224 RGB JPEG 和审计表，并保留官方 train/test 标记。每个数据集运行一次。
02_SOP_Experiments.py：运行嵌套商品级交叉验证实验。通过 --method 选择传统基线或 SupCon。

在 PythonCode 目录、激活 Python 3.11 环境后运行：
python 01_SOP_Preprocessing.py
python 02_SOP_Experiments.py --method baselines
python 02_SOP_Experiments.py --method supcon

01 会在脚本目录及其上级目录查找 Stanford_Online_Products.zip。也可以指定路径：
python 01_SOP_Preprocessing.py --zip-path "D:/data/Stanford_Online_Products.zip"

预处理输出目录非空时不会覆盖。如果 01 已运行完成，且输出目录中已有 clean_manifest.csv、audit_summary.json 和 images/，可以直接设置 data_dir 并运行实验。旧版特征 NPZ 缓存不会复用；首次运行实验时会建立新的特征缓存。

二、依赖
基础依赖包括 numpy、pandas、Pillow、OpenCV、scikit-learn 和 matplotlib。requirements-base.txt 提供可选的 pip 安装列表。已通过 Conda 安装 OpenCV 时无需重复安装其他 OpenCV 版本。
SupCon 和 ResNet 特征需要兼容的 torch/torchvision。CPU 安装示例：
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

安装依赖与 IDE 运行代码时必须使用同一个 Python 解释器。设备参数支持 auto、cpu、cuda、mps；auto 会优先选择可用加速设备，否则使用 CPU。仅 CPU 版 PyTorch 不能使用 NVIDIA GPU。

三、更换特征
在 experiment_config.json 中修改 feature_profile；不需要修改 01 或实验逻辑：
- handcrafted：传统填边 HSV48 与 HSV_HOG192 对比。
- handcrafted_content：只在商品内容区域计算 HSV/HOG，不将填边计入描述符。
- handcrafted_padding_audit：同时计算上述两种方式，并对比填边的影响。
- resnet18_global：冻结的 ImageNet ResNet18 全局池化特征，512 维。
- resnet18：对比 512 维全局特征与额外加入 2×2 局部区域后的 2560 维特征。

内容区域由原始宽高信息还原，不通过检测白色像素估计。HSV 统计和 HOG 网格只在内容区域计算；图像解码和查询图像扰动流程保持一致。ResNet 使用带白边的 224×224 图片输入，并进行 ImageNet 标准化；当前实现不额外执行中心裁剪。

示例配置：
"feature_profile": "resnet18_global"

修改特征后运行 02 即可，无需重跑 01。不同特征配置会使用独立输出目录；相同特征提取和扰动缓存可在传统基线和 SupCon 之间复用。

新增固定特征提取器时，在 sop_feature_bank.py 中实现维数、元数据和 __call__(images)（返回 N×D 浮点数组），并注册到 EXTRACTORS，再在配置中添加特征与列范围。单一特征的 comparisons 使用空列表；对比实验则声明基线与新增特征。若提取器需要用 SOP 标签学习参数，必须在每个训练折内拟合，不能在整个数据集上先行拟合，以免验证集或测试集信息泄漏。

四、共享实验协议
正式默认设置会按完整官方训练商品逐类选择数据，直到纳入至少 10,000 张清理后的原始图片；同一商品的图片全部保留。use_all_official_train 可纳入全部合格训练商品。使用 10 个外层折和 3 个内层折，按商品 ID 划分并手动分层。官方测试集仅作为补充结果，不用于拟合、调参或阈值选择；run_official_test 默认关闭，协议确定后才应启用。

每个抽样锚点配对同商品正样本、同类别随机负样本和同类别颜色困难负样本。困难负样本始终由干净内容区域 HSV48 选择，与实验使用的特征无关。商品折分、图片对清单和校验值在算法及特征配置之间共享；更换特征不会改变评估图片对。

所有方法在干净输入上训练。测试条件仅扰动查询图片，参考图片保持干净。遮挡、噪声、模糊和旋转作用于商品内容区域，白色填边保持不变；随后从受扰图片重新计算特征。

LR/SVM 只使用 |查询特征 − 参考特征|，不拼接逐元素乘积。标准化和 C 值搜索只在内层训练折拟合；先评估 3 个 C 候选，必要时最多两轮扩展。SupCon 依据干净内层验证集匹配 AUC 搜索 3 个温度值；边界最优时可按配置扩展，已评估的候选会复用，AUC 相同时优先选择中间值。所有模型选择仅使用内层验证数据。温度范围耗尽仍在边界时会记录提示。

每个内层及外层训练划分都会独立拟合 SupCon 的标准化器和神经网络。每个外层训练集按类别留出校准商品，比例由 calibration_fraction 设置；校准商品不参与调参、标准化拟合或模型训练。最终模型仅用拟合商品训练一次，再用独立校准商品选择达到 target_validation_recall 的阈值。之后用同一模型评估所有测试条件，不会在校准后重新拟合。有限样本下测试召回率不保证达到校准目标。training_partitions.json 和 calibration.json 会记录划分。

LR/SVM 使用有上限的抽样图片对训练；SupCon 默认每个 epoch 从训练商品中抽样干净图片，除非设置 train_product_cap。评估图片对保持一致，但训练计算量不同，解读结果时应记录预算及 ImageNet 外部预训练情况。仅有一个 512 维特征的配置不能单独衡量空间特征的增益。

五、SupCon 论文适配
参考 Khosla 等人的《Supervised Contrastive Learning》，NeurIPS 2020：
https://arxiv.org/abs/2004.11362
https://github.com/HobbitLong/SupContrast

sop_supcon.py 实现论文公式 (2)：归一化表示、排除自身对比项、使用所有同商品正样本，并在对数概率外对正样本求均值。每批至少包含两个商品，每个商品至少两张不同的干净图片；其他商品提供负样本。

冻结的描述符输入可训练 MLP 编码器和投影头。推理时保留归一化编码器表示用于余弦匹配，丢弃投影头。检查点保存权重、仅由训练数据拟合的标准化器、设置和训练商品；load_model 用于受信任的本地推理。

与论文不同之处：本项目使用固定描述符而非端到端 CNN；使用同商品的不同干净图片而非对同一图片做两种随机增强；任务是未见商品的图片匹配而非 ImageNet 线性分类。因此这是面向本项目的适配，不代表复现论文分类结果。测试扰动不会被用于训练增强。20 个 epoch 是可配置的初始预算，不是论文原始训练日程。

六、配置、输出和续跑
实验配置路径相对于 experiment_config.json 解析；--config 可指定其他配置文件。02 支持 --method、--data-dir、--results-dir、--protocol-dir、--all-train、--skip-official-test 和 --smoke-test。进行方法对比时，公共配置必须相同。

data_dir：处理后的图片和 clean manifest。
cache_dir：按提取器保存的特征数组与元数据。
protocol_dir：共享折分、图片对清单和完整性校验。
results_dir/run_name/feature_profile/baselines 或 supcon：调参记录、逐图片对预测、检查点、折指标、汇总 CSV 和图表。

传统基线和 SupCon 方法都完成后，会生成 combined_summary_mean_sd.csv。paired_feature_deltas.csv 和 paired_delta_mean_sd.csv 保存特征对照结果。grid_requirement_audit.json 标记超参数是否在边界处被选中；只应使用内层验证数据检查该报告。

仅当配置、数据和代码一致时，已完成的外层折才可续跑。更改模型设置时使用新的 run_name；更改数据、采样、折分、条件或协议代码时使用新的 protocol_dir。特征提取器代码变化会产生新的缓存标识。数据集指纹使用 manifest 字节及图片文件大小/修改时间，不是对每张图片内容计算的加密哈希。不要手动修改已保存的图片对 CSV。

版本 3 默认使用 experiment_v3、protocols/shared_v3、feature_cache_v3 和 results_v3。v2 输出属于历史实验记录；不要复用其已完成折或阈值。现有 processed_sop 图片仍兼容，无需重跑 01。更换特征方案时可共享 v3 图片对清单。

七、执行检查
以下 smoke test 使用至少 384 张图片（商品数要求更多时会增加）、2 个外层折、2 个内层折、较小图片对上限和 1 个 SupCon epoch，结果写入独立目录。它们仅检查程序能否执行，不构成正式实验结果。
python 02_SOP_Experiments.py --method baselines --smoke-test
python 02_SOP_Experiments.py --method supcon --smoke-test

运行完整嵌套实验需要较长时间。自动检查覆盖损失函数、梯度、训练折标准化、检查点加载、512 维输入、共享折分和图片对、指标计算及续跑。ResNet 检查用随机权重检查维数和冻结推理，不代表预训练准确率。

历史 notebook 不作为当前配置的运行入口。

八、文件结构
01_SOP_Preprocessing.py 是预处理入口，并直接包含预处理实现。
02_SOP_Experiments.py 通过 --method 选择传统基线或 SupCon。
sop_features.py 包含缩放、HSV/HOG 和 SIFT 辅助函数。
sop_feature_bank.py 包含固定提取器和特征缓存逻辑。
sop_experiment.py 组织共享协议和实验状态。
sop_protocol.py 管理商品折分、图片对生成和协议检查。
sop_baselines.py 与 sop_supcon.py 分别实现两类模型。
sop_calibration.py、sop_tuning.py 和 sop_reporting.py 分别管理阈值校准、调参和结果汇总。
sop_config.py 负责读取和验证配置。