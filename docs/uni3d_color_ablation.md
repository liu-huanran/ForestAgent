# Uni3D Color Ablation Data Preparation

本文档说明如何把当前 mixed Uni3D 输入拆成两个更干净、可解释的 ablation 版本：

- `xyz_only_all`: 所有样本都是 `[10000, 3]`，只含 xyz。
- `color_all`: 所有样本都是 `[10000, 6]`，含 xyz + 明确 policy 生成的 color。

本流程只准备 `.npy` 输入，不运行 Uni3D forward，不重新生成 embedding，不接入主问答系统。

## 1. 为什么当前 mixed 版本难解释

当前历史输入目录 `data/uni3d_all_inputs_rgb` 已经通过 RGB audit round1 检查：

- 430 个 `.npy`
- 279 个是 `[10000, 3]`，即 `xyz_only`
- 151 个是 `[10000, 6]`，带 color 列
- 带 color 的样本中，颜色更像 semantic / discrete / label color，不像 natural RGB
- color availability 与 site_id 强绑定

因此，历史 mixed 版本 `outputs/all_tls_001` 应保留为历史结果，但不适合作为唯一解释依据。它混合了“有颜色”和“无颜色”的输入，random split 下的树种分类结果可能受到 site / color availability / semantic color 的混杂影响。

## 2. xyz_only_all 的意义

`xyz_only_all` 是最干净的 Uni3D 纯几何输入 baseline：

- 所有样本输出 `[10000, 3]`
- 即使 LAS 有 RGB，也丢弃
- 不携带颜色、site palette、species palette 或 semantic color
- 用于检查 Uni3D 仅靠 xyz 几何时的 embedding、similarity 和 downstream probe

推荐输出目录：

```text
data/uni3d_all_inputs_xyz_only_v1
```

## 3. color_all 的意义

`color_all` 用来测试“第 4-6 列 color 通道”对 Uni3D embedding 的影响：

- 所有样本输出 `[10000, 6]`
- 前三列为 xyz
- 后三列为 color，float32 且归一化到 `[0, 1]`
- 每个样本都必须有 color
- color 必须来自明确的 `--color-policy`

推荐输出目录：

```text
data/uni3d_all_inputs_color_all_v1
```

如果不能确认 LAS 中是真实 natural RGB，建议优先用 `constant_all` 作为最干净的 color_all 版本。它会让所有样本都有相同常数 color，不引入额外颜色信息，只测试“模型输入通道统一为 6 维”这一因素。

## 4. Color Policy 风险说明

| color_policy | 用途 | 主要风险 |
|---|---|---|
| `drop` | 仅用于 `xyz_only`，丢弃所有颜色 | 无额外颜色信息，最干净 |
| `constant_all` | 所有样本填同一个常数颜色 | 不携带额外颜色信息，适合作为干净 color_all baseline |
| `las_rgb_if_available_else_constant` | 有 LAS RGB 字段则用 LAS RGB，否则常数 | LAS RGB 字段不等于 natural RGB，必须人工确认字段含义 |
| `existing_semantic_if_available_else_constant` | 沿用旧 `.npy` 中已有 semantic/discrete color，否则常数 | 仍可能混杂 site / semantic color availability |
| `site_palette` | 按 site_id 赋颜色，仅用于诊断 site/domain effect | 明确可能造成 site leakage，不适合正式泛化评估 |
| `species_palette` | 按树种赋颜色，仅用于泄漏演示或诊断 | 明确标签泄漏，不能用于树种分类正式 probe |

注意：不要把当前 semantic/discrete color 称为 natural RGB。

## 5. 生成 xyz_only_all

```bash
python scripts/convert_las_to_uni3d_npy.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/TLS \
  --output-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_all_inputs_xyz_only_v1 \
  --pattern "*.las" \
  --recursive \
  --limit 0 \
  --num-points 10000 \
  --seed 42 \
  --output-mode xyz_only \
  --color-policy drop \
  --skip-existing
```

预期：

- 每个成功样本 `.npy` shape 为 `[10000, 3]`
- `manifest.jsonl` 中 `output_mode=xyz_only`
- `color_policy=drop`
- `has_color_columns=false`
- `natural_rgb=false`
- `semantic_color_likely=false`

## 6. 生成 color_all

### 推荐干净版本：constant_all

```bash
python scripts/convert_las_to_uni3d_npy.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/TLS \
  --output-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_all_inputs_color_all_v1 \
  --pattern "*.las" \
  --recursive \
  --limit 0 \
  --num-points 10000 \
  --seed 42 \
  --output-mode color_all \
  --color-policy constant_all \
  --constant-color 0.4 0.4 0.4 \
  --skip-existing
```

预期：

- 每个成功样本 `.npy` shape 为 `[10000, 6]`
- 后三列都是常数 `[0.4, 0.4, 0.4]`
- `natural_rgb=false`
- `semantic_color_likely=false`
- 这是最适合与 `xyz_only_all` 做干净 ablation 的 color_all 版本

### 诊断版本：沿用旧 semantic color

```bash
python scripts/convert_las_to_uni3d_npy.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/TLS \
  --output-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_all_inputs_color_all_semantic_v1 \
  --pattern "*.las" \
  --recursive \
  --limit 0 \
  --num-points 10000 \
  --seed 42 \
  --output-mode color_all \
  --color-policy existing_semantic_if_available_else_constant \
  --previous-npy-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_all_inputs_rgb \
  --constant-color 0.4 0.4 0.4 \
  --skip-existing
```

这个版本只适合诊断 semantic/discrete color 影响，不适合被表述为 natural RGB 实验。

## 7. 后续 embedding 提取命令模板

不要覆盖历史 `outputs/all_tls_001`。建议新输出名：

```bash
python scripts/batch_extract_uni3d_embeddings.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_all_inputs_xyz_only_v1 \
  --output-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/all_tls_xyz_only_v1 \
  --checkpoint-path /home/team-lu/harrison_workspace/checkpoints/Uni3D/modelzoo/uni3d-b/model.pt \
  --uni3d-repo-path /home/team-lu/harrison_workspace/projects/Uni3D \
  --device cuda \
  --pc-model eva02_base_patch14_448 \
  --pc-feat-dim 768 \
  --embed-dim 1024 \
  --pc-encoder-dim 512 \
  --num-group 512 \
  --group-size 64 \
  --recursive \
  --limit 0 \
  --skip-existing
```

将 `--input-dir` 和 `--output-dir` 换成 color_all 版本即可：

```text
input:  /home/team-lu/harrison_workspace/data/forestagent/uni3d_all_inputs_color_all_v1
output: /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/all_tls_color_all_v1
```

## 8. 后续 similarity analysis

```bash
python scripts/analyze_uni3d_embedding_similarity.py \
  --embedding-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/all_tls_xyz_only_v1 \
  --output-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/all_tls_xyz_only_v1_analysis \
  --top-k 5 \
  --save-full-matrix
```

color_all 版本同理替换目录。

## 9. 后续 strict downstream probe

```bash
python scripts/probe_uni3d_embeddings_strict.py \
  --embedding-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/all_tls_xyz_only_v1 \
  --label-path /home/team-lu/harrison_workspace/projects/ForestAgent/data/parameters.xlsx \
  --output-dir /home/team-lu/harrison_workspace/outputs/forestagent/probe_strict/all_tls_xyz_only_v1_loso \
  --tasks all \
  --feature-sets all \
  --split leave_one_site_out \
  --overwrite
```

推荐比较顺序：

1. mixed current: `outputs/all_tls_001`
2. xyz_only_all: `all_tls_xyz_only_v1`
3. color_all constant: `all_tls_color_all_v1`

downstream probe 必须优先看 group split / leave-one-site-out。不要只看 random split。

## 10. 不能作为论文结论的内容

- 不能把 species_palette 结果作为正式树种分类结果。
- 不能把 site_palette 结果作为跨 site 泛化能力。
- 不能把 semantic/discrete color 说成 natural RGB。
- 不能只用 random split 说明 Uni3D 对树种有跨域语义能力。
- 不能把 ablation embedding 结果直接等同于主问答系统提升。

## 11. 不要提交 Git 的内容

不要提交：

- `data/`
- `outputs/`
- `checkpoints/`
- `weights/`
- `*.npy`
- `*.npz`
- `*.las`
- `*.laz`
- `*.pt`
- `*.pth`
- `*.ckpt`

