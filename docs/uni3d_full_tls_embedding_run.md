# Uni3D Full TLS Embedding Run

本文档用于把 Uni3D TLS 数据处理流程从 10-tree sanity check 扩展到可安全跑全量 TLS 单木数据的工程准备阶段。

## 1. 当前阶段目标

- 从 10 棵树 sanity check 扩大到 TLS 全量 `.las` 单木数据的 Uni3D embedding 提取。
- 目标是生成可追踪、可续跑的全量 `.npy` 输入、`.npz` embedding 缓存、summary 和日志。
- 仍不接入 agent。
- 仍不接入主问答系统。
- 仍不修改 q1/q2/q3 geometry baseline。
- 仍不把结果包装成论文结论。

已知 10-tree sanity check 通过：

- `success_count = 10`
- `failure_count = 0`
- `embedding_dim = 1024`
- `invalid_count = 0`
- pairwise cosine similarity min / mean / max = `0.6335 / 0.8661 / 0.9688`

这个结果只说明小批量链路可用，不等价于全量实验结论。

## 2. 推荐运行顺序

```text
服务器环境检查
-> LAS -> NPY 全量转换
-> 检查转换 summary
-> Uni3D embedding 全量提取
-> 检查 extraction summary
-> 抽样 similarity 分析
-> 视情况做全量 similarity 或 top-k 分析
```

## 3. 服务器环境检查

```bash
cd /home/team-lu/harrison_workspace/projects/ForestAgent
conda activate fa-u
export CUDA_VISIBLE_DEVICES=1

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import laspy; print('laspy ok')"
python -c "import numpy; print('numpy ok')"
python -c "import pointnet2_ops; print('pointnet2_ops ok')"
python -c "from forestagent.adapters.uni3d_extractor import Uni3DFeatureExtractor; print('ForestAgent Uni3DFeatureExtractor ok')"
python -c "import sys; sys.path.insert(0, '/home/team-lu/harrison_workspace/projects/Uni3D'); import models.uni3d; print('Uni3D models.uni3d ok')"
```

如果 `laspy` 缺失：

```bash
conda activate fa-u
pip install laspy
```

## 4. LAS -> NPY 全量转换

输出目录建议放在 server data 区，不要放进 Git 工作区跟踪文件列表：

```bash
cd /home/team-lu/harrison_workspace/projects/ForestAgent
conda activate fa-u

python scripts/convert_las_to_uni3d_npy.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/TLS \
  --output-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_tls_npy_10000 \
  --pattern "*.las" \
  --recursive \
  --limit 0 \
  --num-points 10000 \
  --seed 42 \
  --skip-existing
```

如果要尝试 LAS RGB：

```bash
python scripts/convert_las_to_uni3d_npy.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/TLS \
  --output-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_tls_npy_10000_rgb \
  --pattern "*.las" \
  --recursive \
  --limit 0 \
  --num-points 10000 \
  --seed 42 \
  --include-rgb \
  --skip-existing
```

转换输出：

- 每个成功样本一个 `.npy`，shape 为 `[10000, 3]` 或 `[10000, 6]`，dtype 为 `float32`。
- `manifest.jsonl`：每行记录 `tree_id`、`source_las`、`output_npy`、`status`、`failure_reason`、`input_point_count`、`output_point_count`、`output_shape`、`has_rgb`、`include_rgb`、`sampling_method`、`seed`。
- `summary.json`：记录 `candidate_count`、`success_count`、`skipped_count`、`failure_count` 和逐样本结果。

## 5. 检查转换 summary

```bash
python - <<'PY'
import json
from pathlib import Path

summary_path = Path('/home/team-lu/harrison_workspace/data/forestagent/uni3d_tls_npy_10000/summary.json')
summary = json.loads(summary_path.read_text())
print('candidate_count =', summary['candidate_count'])
print('success_count =', summary['success_count'])
print('skipped_count =', summary['skipped_count'])
print('failure_count =', summary['failure_count'])
for item in summary['results'][:5]:
    print(item['tree_id'], item['status'], item['output_shape'], item['failure_reason'])
PY
```

如果 `failure_count` 很大，先不要跑 GPU embedding。优先看是否大量点数少于 `10000`，或 `.las` 目录是否选错。

## 6. Uni3D embedding 全量提取

batch extraction 的 `--input-dir` 指向上一步转换脚本的 `--output-dir`：

```bash
cd /home/team-lu/harrison_workspace/projects/ForestAgent
conda activate fa-u
export CUDA_VISIBLE_DEVICES=1

python scripts/batch_extract_uni3d_embeddings.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_tls_npy_10000 \
  --output-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/tls_full_10000 \
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
  --pattern "*.npy" \
  --skip-existing
```

提取输出：

- 每个成功样本一个 `.npz`，包含 `embedding_raw`、`embedding_l2`、`tree_id`、`source_path`、`metadata`。
- `summary.json`，记录 `candidate_count`、`success_count`、`skipped_count`、`failure_count`、embedding shape、finite 检查和 L2 norm。

## 7. 检查 extraction summary

```bash
python - <<'PY'
import json
from pathlib import Path

summary_path = Path('/home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/tls_full_10000/summary.json')
summary = json.loads(summary_path.read_text())
print('candidate_count =', summary['candidate_count'])
print('success_count =', summary['success_count'])
print('skipped_count =', summary['skipped_count'])
print('failure_count =', summary['failure_count'])
bad = [item for item in summary['results'] if item['status'] == 'failed']
print('first failures =', bad[:5])
PY
```

## 8. 抽样 similarity 分析

全量 pairwise similarity 可以计算，但不建议第一步默认保存完整 NxN CSV。先做抽样：

```bash
cd /home/team-lu/harrison_workspace/projects/ForestAgent
conda activate fa-u

python scripts/analyze_uni3d_embedding_similarity.py \
  --embedding-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/tls_full_10000 \
  --output-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/tls_full_10000_similarity_sample \
  --max-samples 500 \
  --seed 42 \
  --top-k 5 \
  --no-full-matrix
```

如果样本数不大，或你明确需要完整 CSV：

```bash
python scripts/analyze_uni3d_embedding_similarity.py \
  --embedding-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/tls_full_10000 \
  --output-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/tls_full_10000_similarity_full \
  --top-k 5 \
  --save-full-matrix
```

默认策略：

- 样本数 `<= 1000` 时默认保存 `cosine_similarity.csv`。
- 样本数 `> 1000` 时默认不保存完整 NxN CSV，只保存 `similarity_summary.json` 和 `nearest_neighbors.json`。
- 可用 `--no-full-matrix` 强制不保存。
- 可用 `--save-full-matrix` 强制保存。

## 9. 中断后续跑

转换中断后续跑：

```bash
python scripts/convert_las_to_uni3d_npy.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/TLS \
  --output-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_tls_npy_10000 \
  --pattern "*.las" \
  --recursive \
  --limit 0 \
  --num-points 10000 \
  --seed 42 \
  --skip-existing
```

embedding 提取中断后续跑：

```bash
python scripts/batch_extract_uni3d_embeddings.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_tls_npy_10000 \
  --output-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/tls_full_10000 \
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
  --pattern "*.npy" \
  --skip-existing
```

不要使用 `--overwrite`，除非你明确要重新生成对应阶段全部输出。

## 10. 通过标准

可以认为工程链路阶段通过：

- 大部分 `.las` 成功转换成 `.npy`，失败样本原因明确。
- 大部分 `.npy` 成功生成 `.npz` embedding，失败样本原因明确。
- `embedding_l2` finite。
- `embedding_l2` L2 norm 接近 1。
- `invalid_count` 为 0 或极少，且原因明确。
- pairwise similarity 不是全部接近 1。
- summary 和 manifest 足够定位每一个失败样本。

## 11. 不能算通过的情况

- 大量 LAS 转换失败。
- 大量点数不足。
- checkpoint load 失败。
- `state_dict` shape mismatch。
- 大量 Uni3D forward 失败。
- 大量 embedding 出现 NaN / Inf。
- 输出 embedding 全部近似重复。
- 用 mock 或 fake embedding 代替真实结果。

## 12. 不要提交到 Git 的文件

不要提交：

- `data/`
- `outputs/`
- `checkpoints/`
- `weights/`
- `*.pt`
- `*.pth`
- `*.ckpt`
- 大规模 `.npy`
- 大规模 `.npz`
- 大规模 similarity matrix CSV

这些已经基本被 `.gitignore` 覆盖，但提交前仍要用 `git status --short` 复查。
