# Uni3D Downstream Probe

本文档记录如何基于已经提取好的冻结 Uni3D embedding 做严格的 offline downstream probe。

## 1. 当前阶段目标

本阶段只回答一个很窄的问题：

`precomputed Uni3D embedding -> lightweight probe -> 是否存在下游预测信号`

它不是 Uni3D 训练脚本，也不是主问答系统集成。

当前允许做：

- 读取已有 `.npz` embedding。
- 读取 `data/parameters.xlsx` 标签表。
- 对树种分类、胸径、树高、东西冠幅、南北冠幅做轻量 probe。
- 使用 random / stratified_random / group / leave-one-site-out split 检查泛化。
- 输出结果表、预测表、confusion matrix 和 warning。

当前不允许做：

- 不重新运行 Uni3D forward。
- 不训练或微调 Uni3D backbone。
- 不接入 agent / LLM / 主问答路径。
- 不修改 q1/q2/q3 geometry baseline。
- 不把 probe 结果写成论文最终结论。

## 2. 数据准备

输入一：embedding 目录。

目录中每棵树一个 `.npz`，例如：

```text
outputs/all_tls_001/
  001_150_100.npz
  002_150_101.npz
  ...
```

脚本默认读取 `embedding_l2`，并接受 `[1024]` 或 `[1, 1024]`，内部会 squeeze 成 `[1024]`。

输入二：标签表。

当前本地侦察到 `data/parameters.xlsx` 包含 430 行，列名包括：

```text
样地号
树种
对应的文件名
编号
X
Y
Z
胸径cm
树高m
东西冠幅m
南北冠幅m
```

默认 tree_id 对齐规则：

- 标签表用 `对应的文件名`，如 `150_64.las`。
- 去掉 `.las` 后得到 `150_64`。
- embedding 内部 `tree_id` 或文件名推断出的 `tree_id` 也应是 `150_64`。
- 不做危险模糊匹配；无法确定就记录 unmatched。

## 3. 为什么需要 Train/Test Split

全量 embedding 提取成功只说明 feature cache 可以生成。

一旦做 supervised probe，就必须拆分 train/test：

- scaler 只能在 train 上 fit。
- 模型只能在 train 上 fit。
- test 只能 transform 和 predict。
- 全量训练再全量测试只能作为 exploratory，不应进入正式结果主表。

## 4. Split 说明

`random`：

- 宽松参考。
- 多 seed 重复。
- 同一个 site 可能同时出现在 train/test，因此不能代表跨样地泛化。

`stratified_random`：

- 仅用于树种分类。
- 尽量保持 train/test 的树种比例。
- 类别样本过少时会跳过并记录原因。

`group`：

- 默认按 `tree_id` 前缀提取 site_id，例如 `160_76 -> 160`。
- 不允许同一个 site 同时出现在 train/test。

`leave_one_site_out`：

- 每次留一个 site 做 test，其余 site 做 train。
- 当前最重要的严格泛化检查之一。
- 如果 random 高但 leave-one-site-out 明显下降，要警惕 site/domain leakage。

## 5. 防泄漏说明

脚本支持三类 feature set：

- `uni3d`：只用冻结 Uni3D embedding。
- `geometry`：只用标签表中的几何字段。
- `uni3d_plus_geometry`：拼接 Uni3D embedding 和允许使用的几何字段。

回归任务的硬性防泄漏规则：

- 预测胸径时，geometry input 不能包含 `胸径cm`。
- 预测树高时，geometry input 不能包含 `树高m`。
- 预测东西冠幅时，geometry input 不能包含 `东西冠幅m`。
- 预测南北冠幅时，geometry input 不能包含 `南北冠幅m`。

其他混杂因素：

- near-duplicate 可能抬高 random split 结果。
- site 分布可能造成 domain leakage。
- `rgb_source=provided` 与 `fallback_constant` 的分布差异可能造成 RGB availability confound。

## 6. 命令示例

### Random Split

```bash
python scripts/probe_uni3d_embeddings_strict.py \
  --embedding-dir outputs/all_tls_001 \
  --label-path data/parameters.xlsx \
  --output-dir outputs/probe_strict/random_round1 \
  --tasks all \
  --feature-sets all \
  --split random \
  --seeds 0 1 2 3 4 \
  --overwrite
```

### Leave-One-Site-Out

```bash
python scripts/probe_uni3d_embeddings_strict.py \
  --embedding-dir outputs/all_tls_001 \
  --label-path data/parameters.xlsx \
  --output-dir outputs/probe_strict/loso_round1 \
  --tasks all \
  --feature-sets all \
  --split leave_one_site_out \
  --overwrite
```

### Exclude Near-Duplicates

```bash
python scripts/probe_uni3d_embeddings_strict.py \
  --embedding-dir outputs/all_tls_001 \
  --label-path data/parameters.xlsx \
  --output-dir outputs/probe_strict/no_near_duplicates \
  --tasks all \
  --feature-sets all \
  --split all \
  --near-duplicate-path outputs/all_tls_001_embedding_analysis_round1/near_duplicate_pairs.csv \
  --exclude-near-duplicates \
  --overwrite
```

### RGB Provided Only

```bash
python scripts/probe_uni3d_embeddings_strict.py \
  --embedding-dir outputs/all_tls_001 \
  --label-path data/parameters.xlsx \
  --output-dir outputs/probe_strict/rgb_provided_only \
  --tasks all \
  --feature-sets all \
  --split all \
  --review-table-path outputs/all_tls_001_embedding_quant_review_round1/review_table.csv \
  --rgb-source provided \
  --overwrite
```

## 7. 输出文件

脚本在 `--output-dir` 中写出：

- `label_inventory.json`：标签表列名、自动识别列、非空数量、dtype、示例值。
- `alignment_report.json`：embedding/label 对齐数量、unmatched、缺失标签计数、过滤统计。
- `probe_results.csv`：每个 task / feature_set / split / metric 一行。
- `probe_summary.json`：聚合结果、warning、leave-one-site-out 分站点结果。
- `per_split_predictions.csv`：每个 test 样本的 y_true / y_pred。
- `per_class_metrics.csv`：分类任务每类 precision / recall / f1 等。
- `confusion_matrices/`：分类任务每个 split 的混淆矩阵。
- `run_config.json`：本次运行参数、输入输出路径、embedding key 和过滤条件。

## 8. 如何解释结果

推荐优先看：

- `uni3d` vs `geometry` vs `uni3d_plus_geometry`。
- `random` vs `leave_one_site_out`。
- 排除 near-duplicate 前后的变化。
- `rgb_source=provided` 与 `fallback_constant` 分层结果。

解释边界：

- 可以说：冻结 Uni3D embedding 在某个严格 split 下表现为某种指标。
- 不要说：Uni3D 已经正式提升主系统。
- 不要说：Uni3D 可以替代 q1/q2/q3 几何工具。
- 不要说：probe 结果就是论文最终结论。

## 9. 不能提交 Git 的内容

不要提交：

- `data/`
- `outputs/`
- `checkpoints/`
- `weights/`
- 大规模 `.npy`
- 大规模 `.npz`
- `*.pt`
- `*.pth`
- `*.ckpt`

