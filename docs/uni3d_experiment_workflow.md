# Uni3D 离线实验编排工作流

这份文档说明 ForestAgent 中新增的 Uni3D 离线实验管理层。它只负责实验配置、命令计划、结果审计、横向对比和 Markdown 记录，不是主问答系统的一部分。

## 1. 为什么需要这个框架

当前 Uni3D 工作已经包含多条离线路径：

- LAS 到 Uni3D `.npy` 输入转换
- 冻结 Uni3D embedding 提取
- cosine similarity 分析
- strict downstream probe
- RGB / semantic color 审计
- mixed / xyz-only / color-all ablation 对比

如果每次都靠人工拼命令，很容易忘记参数、覆盖旧结果或误读实验边界。实验框架的目标是把一次实验写成配置文件，再由工具生成命令、审计输出、比较多个版本，并生成可复制到实验记录里的 Markdown。

## 2. 当前边界

这套流程是离线工程工具层。

- 不自动运行 Uni3D forward。
- 不训练 Uni3D。
- 不把 Uni3D 注册成 agent 工具。
- 不修改 q1 / q2 / q3 geometry baseline。
- 不修改主问答路径。
- 不把 probe 或 similarity 结果包装成论文最终结论。

## 3. 实验版本命名

推荐使用：

- `all_tls_mixed_legacy_v1`：历史 mixed 输入版本，对应旧的 `all_tls_001` artifact。
- `all_tls_xyz_only_v1`：干净的纯几何输入版本，所有样本为 `[10000, 3]`。
- `all_tls_color_all_v1`：统一颜色通道版本，所有样本为 `[10000, 6]`，但颜色来源必须由 `color_policy` 明确记录。

注意：`color_all` 不等于 natural RGB。当前证据提示部分颜色更像 semantic / discrete color，因此解释时必须谨慎。

## 4. 推荐目录结构

服务器示例：

```text
/home/team-lu/harrison_workspace/data/forestagent/
  tls_las/
  uni3d_all_inputs_xyz_only_v1/
  uni3d_all_inputs_color_all_v1/

/home/team-lu/harrison_workspace/outputs/forestagent/
  uni3d_embeddings/all_tls_xyz_only_v1/
  uni3d_embeddings/all_tls_xyz_only_v1_analysis/
  uni3d_embeddings/all_tls_color_all_v1/
  uni3d_embeddings/all_tls_color_all_v1_analysis/
  probe_strict/all_tls_xyz_only_v1/
  probe_strict/all_tls_color_all_v1/
```

本地生成的小报告建议放在：

```text
outputs/local_reports/
```

大规模数据、embedding、checkpoint 和输出结果不要提交到 Git。

## 5. Experiment Config

配置文件位于：

```text
configs/uni3d_experiments/
```

每个配置记录：

- 实验名、版本、阶段和风险说明
- 原始 LAS、转换后 NPY、embedding、similarity、probe 输出路径
- checkpoint 和官方 Uni3D 仓库路径
- conversion 的 `output_mode` 和 `color_policy`
- extractor 模型参数
- similarity 和 probe 设置

示例配置中的路径是服务器风格占位路径。真正运行前需要按实验室服务器实际路径确认和修改。

## 6. 生成命令计划

只生成命令，不执行：

```bash
python scripts/plan_uni3d_experiment.py \
  --config configs/uni3d_experiments/all_tls_xyz_only_v1.yaml \
  --output-plan outputs/local_reports/experiment_plans \
  --stage all \
  --print
```

输出：

- `outputs/local_reports/experiment_plans/<experiment>_commands.sh`
- `outputs/local_reports/experiment_plans/<experiment>_plan.json`

生成的 `.sh` 是命令计划，不是自动实验系统。长时间 GPU 任务仍然建议你在服务器 `tmux` 中手动执行。

## 7. 在服务器执行

示例：

```bash
tmux new -s uni3d_xyz_v1
conda activate fa-u
cd ~/harrison_workspace/projects/ForestAgent
bash outputs/local_reports/experiment_plans/all_tls_xyz_only_v1_commands.sh
```

如果希望使用 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1 bash outputs/local_reports/experiment_plans/all_tls_xyz_only_v1_commands.sh
```

## 8. 审计实验输出

示例：

```bash
python scripts/audit_uni3d_experiment.py \
  --config configs/uni3d_experiments/all_tls_mixed_legacy_v1.yaml \
  --output-dir outputs/local_reports/experiment_audits/all_tls_mixed_legacy_v1 \
  --print
```

审计是只读的，会检查：

- conversion：`.npy` 数量、shape、dtype、manifest、summary
- extraction：`.npz` 数量、`summary.json`、embedding shape、finite、L2 norm、mock metadata
- similarity：`similarity_summary.json`、nearest neighbors、near duplicate、outlier
- probe：`probe_results.csv`、`probe_summary.json`、split 类型、feature set、prediction、alignment 文件

缺少某个阶段时会记录为 `NOT_AVAILABLE`，不会脑补为成功。

## 9. 对比多个实验

示例：

```bash
python scripts/compare_uni3d_experiments.py \
  --configs \
    configs/uni3d_experiments/all_tls_mixed_legacy_v1.yaml \
    configs/uni3d_experiments/all_tls_xyz_only_v1.yaml \
    configs/uni3d_experiments/all_tls_color_all_v1.yaml \
  --output-dir outputs/local_reports/experiment_comparisons/mixed_xyz_color_v1 \
  --print
```

输出：

- `experiment_comparison_summary.json`
- `experiment_comparison_table.csv`
- `experiment_comparison_report.md`

对比重点：

- 输入 shape 与颜色策略
- embedding 成功率、维度、finite、L2 norm
- similarity min / mean / max / std
- near duplicate / outlier 数量
- strict probe 中 `uni3d`、`geometry`、`uni3d_plus_geometry` 的差异
- random split 与 leave-one-site-out 的差异

## 10. 生成实验报告

示例：

```bash
python scripts/generate_uni3d_experiment_report.py \
  --config configs/uni3d_experiments/all_tls_xyz_only_v1.yaml \
  --audit-json outputs/local_reports/experiment_audits/all_tls_xyz_only_v1/experiment_audit_report.json \
  --output-md outputs/local_reports/experiment_reports/all_tls_xyz_only_v1_report.md \
  --memory-snippet-output outputs/local_reports/experiment_reports/all_tls_xyz_only_v1_memory_snippet.md
```

`project_memory_snippet.md` 只是候选片段，需要人工看过后再复制进 `PROJECT_MEMORY.md`，不会自动修改项目记忆。

## 11. 如何解释 mixed / xyz-only / color-all

`all_tls_mixed_legacy_v1` 只能作为历史参考，因为 RGB/color availability 曾经是混合的，并且和 site_id 存在混杂风险。

`all_tls_xyz_only_v1` 是最干净的纯几何 Uni3D embedding baseline。

`all_tls_color_all_v1` 必须看清 `color_policy`：

- `constant_all`：最干净的通道数对照，不携带额外颜色信息。
- `existing_semantic_if_available_else_constant`：可能保留 semantic/discrete color 混杂。
- `site_palette`：只适合诊断 site/domain effect，有 site leakage 风险。
- `species_palette`：标签泄漏，不能用于正式树种分类 probe。

## 12. 为什么不能只看 random split

random split 可能让同一个 site 同时出现在 train/test。对于 TLS 单木数据，这可能让 site、采集风格、semantic color 或 near duplicate 抬高结果。

更重要的评估是：

- group split by site
- leave-one-site-out

如果 random split 高、leave-one-site-out 低，应优先怀疑 site/domain leakage，而不是直接说 Uni3D 特征有效。

## 13. 不要提交的文件

不要提交：

- `data/`
- `outputs/`
- `checkpoints/`
- `weights/`
- `*.pt`
- `*.pth`
- `*.ckpt`
- 大规模 `*.npy`
- 大规模 `*.npz`
- LAS / LAZ 原始点云

只提交源码、配置、文档和小型测试。

## 14. 推荐工作循环

1. 写或复制一个 experiment config。
2. 生成命令计划。
3. 在 GPU 服务器手动执行命令计划。
4. 必要时只同步 summary/report，不同步大数据。
5. 审计 conversion / extraction / similarity / probe 输出。
6. 比较 mixed / xyz-only / color-all。
7. 生成 Markdown 实验记录和 PROJECT_MEMORY 候选片段。
8. 只把已确认事实写进 `PROJECT_MEMORY.md`。

