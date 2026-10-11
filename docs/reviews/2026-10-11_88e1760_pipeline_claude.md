# 可配置流程（01/02/03 与 SupCon）互评

- 被评审提交：`88e1760`（xingyuqiong，feat: add configurable SOP pipeline and SupCon matching）。
- 评审者：本会话 Claude AI（由 zyFaye 使用）；不代表其他组员或独立评审者同意。
- 日期：2026-10-11（Australia/Melbourne）。
- 文件与范围：`PythonCode/` 下的 `sop_config.py`、`sop_runner.py`、`sop_preprocessing.py`、`sop_protocol.py`、`sop_feature_bank.py`、`sop_experiment.py`、`sop_baselines.py`、`sop_supcon.py`、`sop_reporting.py`、`tests/test_pipeline.py`、`Readme.txt`、`experiment_config.json`。
- 依据：[当前方案](../current-plan.md)、[官方要求](../../materials/official/Project.md)、[组员回应](2026-10-11-configurable-pipeline-response.md)、[VALIDATION](../../PythonCode/VALIDATION.txt)。
- 验证方式：
  1. 全部源码静态阅读。
  2. 本会话临时虚拟环境（macOS，Python 3.11.15，torch 2.14.1+cpu，torchvision 0.29.1，numpy 2.4.6，pandas 3.0.6，scikit-learn 1.9.1，OpenCV 4.14.0，Pillow 12.3.0，matplotlib 3.11.2）运行 `cd PythonCode && python -m unittest discover -s tests -v`：5 项全部通过，10.1 秒。
  3. 合成数据探针：手写 ROC-AUC 与逐对暴力计算一致（0.761693）；`recall_threshold` 在 95% 目标下验证集 TPR 为 0.953；干净特征缓存与从保存 JPEG 重新提取完全一致（最大差 0.0）。
- 未覆盖：真实 SOP 数据、`--smoke-test` 实跑（本机无 `processed_sop/`）、Windows、GPU/MPS、预训练 ResNet 权重及其性能。

## 已确认的优点

- 商品级隔离：外层与内层都检查训练/测试商品无交集（`sop_experiment.py` 135–136、`sop_supcon.py` 226–227）；官方 test 不参与调参和阈值选择。
- 图片对与特征解耦：困难负例固定用干净 HSV48 选择，换特征不换图片对；图片对 CSV 带 SHA-256 校验，篡改会报错（测试已覆盖）。
- 标准化、C/温度选择、阈值均只用相应训练部分；SupCon 每个内层/外层训练部分重新训练，推理丢弃投影头（测试已覆盖）。
- 指标手写实现，报告外层折均值与样本标准差；断点复用要求配置、数据、代码一致。
- 旧评审 R2（JPEG 缓存不一致）已修复，见 [预处理互评](2026-10-11_3054cd3_preprocessing-features_claude.md)。

## R1：LR/SVM 图片对特征与说明不一致

- 类型/优先级：确认问题；正式实验前需处理，影响方法描述和 LR/SVM 的表达能力。
- 状态：待回应
- 位置与触发条件：`PythonCode/Readme.txt:74` 写明 “LR/SVM use absolute difference plus elementwise product of image features”；实际 `sop_protocol.pair_matrix`（155–158 行）只返回 `|query - reference|`。所有 LR/SVM 运行都会触发。
- 证据：合成探针中 HSV48 的图片对特征维度为 48；若包含逐元素乘积应为 96。
- 影响：报告若按 Readme 描述方法会与代码不符。仅用绝对差时，线性 LR 相当于学习加权 L1 距离，无法利用“两图在同一维度都很高”这类共现信息；是否需要乘积项属于方法选择，不是对错问题。
- 方案 A（改说明）：保留现有实现，将 Readme 改为 “absolute difference only”，报告中如实描述。代价最小，已有合成结果不变。
- 方案 B（改代码）：`pair_matrix` 拼接 `|q - r|` 与 `q * r`，维度翻倍。表达力更强，但 RBF SVM 与 LR 的成本上升，ResNet 2560 维时输入达 5120 维；修改后需新的 `run_name`。
- 建议与验证方法：由负责人决定 A 或 B，并与 Readme、报告描述保持一致；在合成数据上断言 `pair_matrix` 输出维度符合说明。
- 负责人回应：待回应
- 实施者/提交：尚未实施
- 复查结果：尚未验证

## R2：官方 holdout 查询依赖两处独立抽样恰好一致

- 类型/优先级：待验证风险（当前代码下结果正确）；中等，未来修改时可能静默出错。
- 状态：待回应
- 位置与触发条件：扰动特征只为 `query_mask` 中的图片计算，其余行直接复制干净特征（`sop_feature_bank.py` 132–133 行）。官方 holdout 的 `query_mask` 由 `sop_experiment.py` 86–89 行用 `default_rng(seed + 50002).permutation(products)[:evaluation_anchor_cap]` 抽取；实际测试查询则由 `make_pairs`（`sop_protocol.py` 118 行）用同一种子、同一上限独立再抽一次。两者目前完全相同，但代码中没有断言保证。
- 证据：合成探针中，当前实现的官方查询全部在 `query_mask` 内；若只把一侧的上限改小 4，就有 4 个查询商品不在覆盖范围内，而流程不会报错。
- 影响：一旦有人只改其中一处（种子偏移、上限、抽样方式），部分查询的“扰动特征”会等于干净特征，官方 holdout 的鲁棒性结果偏乐观且无提示。外层十折的 `query_mask` 覆盖全部所选训练商品，不受影响。
- 方案 A（最小改动）：在 `Experiment.context` 或 `make_pairs` 返回后断言 `test_pairs` 的查询图片都在 `self.query_mask` 中，否则报错。
- 方案 B：先生成官方测试图片对，再由其查询商品反推 `query_mask`，去掉重复抽样。
- 建议与验证方法：推荐方案 A；加一个合成测试，故意让两侧上限不同，确认会报错。
- 负责人回应：待回应
- 实施者/提交：尚未实施
- 复查结果：尚未验证

## R3：SupCon 温度选在边界时不自动扩展

- 类型/优先级：已知限制（组员回应中已说明）；正式实验后需检查。
- 状态：待回应
- 位置：LR/SVM 的 C 在边界胜出时最多自动扩展两次网格（`sop_baselines.py` 65–70 行）；SupCon 只记录 `boundary_unresolved`（`sop_supcon.py` 240–246 行），不扩展温度范围。
- 证据：本次合成测试中 SupCon/HSV48 在 2 个外层折中有 1 个选到边界，报告打印 “Hyperparameter range needs review”。合成数据不说明真实情况，只证明审计提示生效。
- 影响：[当前方案](../current-plan.md) 要求“边界值经常胜出则扩展范围”。若真实运行中温度经常落在 0.05 或 0.2，需要扩展网格并重跑，不能只报告提示。
- 方案 A：保持现状，正式运行后人工查看 `grid_requirement_audit.json`，必要时改网格并换 `run_name` 重跑。
- 方案 B：仿照 LR/SVM 增加有上限的自动扩展（如向两侧各乘/除 2）。代价是训练次数增加。
- 负责人回应：待回应
- 实施者/提交：尚未实施
- 复查结果：尚未验证

## R4：较小的一致性问题

- 类型/优先级：可选改进；低。
- 状态：待回应
- [docs/data.md](../data.md) 第 24 行仍写预处理会保存 `.npz` 特征；新 01 已不生成。
- `sop_preprocessing.py:16` 仍在 `notebooks/` 中查找 `sop_features.py`，是旧代码残留；若同目录缺文件，可能误用 `notebooks/` 中的旧副本。
- 根目录 `requirements-base.txt` 与 `PythonCode/requirements-base.txt` 是两份不同的依赖清单，后续容易不一致。
- 默认 `run_official_test: true` 会为全部约 6 万张官方 test 图提取干净特征；ResNet 在 CPU 上耗时较长。开发阶段可按 Readme 使用 `--skip-official-test`。
- 负责人回应：待回应

## 结论

- 流程结构、商品隔离、调参边界和手写指标在本次范围内未发现泄漏；5 项合成测试在 Mac/Python 3.11 上复现通过。
- 正式运行前建议先处理 R1（统一说明与实现）和 R2（加断言）；R3 在正式运行后按审计结果处理。
- 合成测试只证明代码可执行，不是可报告的实验结果；真实 SOP、Windows 与预训练 ResNet 仍待验证。
- 本记录只评价，不修改被评审代码。
