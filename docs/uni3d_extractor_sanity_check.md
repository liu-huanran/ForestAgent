# Uni3D Extractor Sanity Check

本文档记录当前阶段如何在服务器上验证 ForestAgent 的独立 Uni3D extractor。

## 1. 当前阶段目标

本阶段只做小批量 sanity check：

- 验证 `Uni3DFeatureExtractor` 能在真实 GPU 环境下对多棵单木点云提取 global embedding。
- 检查每棵树的 embedding shape、finite 状态、L2 norm、metadata。
- 读取 5-10 棵树的 embedding，计算两两 cosine similarity，初步判断是否有基本区分度。
- 不做正式任务效果评估，不证明 Uni3D 一定改善 q1/q2/q3。
- 不接入主问答系统，不注册 agent tool，不训练模型。

## 2. 服务器前置条件

运行前需要确认：

- GPU 可见，并且建议用 `CUDA_VISIBLE_DEVICES=1` 指定 GPU 1。
- 已进入 conda 环境 `fa-u`。
- PyTorch CUDA 版本可用。
- 官方 Uni3D 仓库已在服务器本地，例如 `/home/team-lu/harrison_workspace/projects/Uni3D`。
- `pointnet2_ops` 可以 import。
- Uni3D checkpoint 已存在，例如 `/home/team-lu/harrison_workspace/checkpoints/Uni3D/modelzoo/uni3d-b/model.pt`。
- 已准备小样本单木点云 `.npy`，每个文件 shape 为 `[N, 3]` 或 `[N, 6]`。

当前 batch 脚本第一版只直接支持 `.npy`：

- `[N, 3]` 表示 xyz，RGB 使用 adapter 的 `rgb_fallback` 常数。
- `[N, 6]` 表示 xyzrgb，前三列是 xyz，后三列是 rgb。
- `.las` 需要先在服务器上转换成 `.npy`，本脚本暂不内置 `.las` 读取，避免引入额外依赖和坐标约定。

## 3. 服务器检查命令

```bash
nvidia-smi
conda activate fa-u

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import timm; print('timm ok')"
python -c "import easydict; print('easydict ok')"
python -c "import numpy; print('numpy ok')"
python -c "import laspy; print('laspy ok')"
python -c "import pointnet2_ops; print('pointnet2_ops ok')"

cd /home/team-lu/harrison_workspace/projects/ForestAgent
python -c "from forestagent.adapters.uni3d_extractor import Uni3DFeatureExtractor; print('ForestAgent Uni3DFeatureExtractor import ok')"

cd /home/team-lu/harrison_workspace/projects/Uni3D
python -c "import models.uni3d; print('official Uni3D models.uni3d import ok')"
```

如果使用 GPU 1：

```bash
export CUDA_VISIBLE_DEVICES=1
```

## 4. LAS 转 NPY 数据准备流程

完整小批量流程是：

```text
raw .las -> converted .npy -> batch extraction -> similarity analysis
```

`batch_extract_uni3d_embeddings.py` 的核心职责仍然只读取 `.npy` 并调用 Uni3D extractor。`.las` 读取和固定采样由独立脚本 `scripts/convert_las_to_uni3d_npy.py` 完成。

检查 `laspy`：

```bash
conda activate fa-u
python -c "import laspy; print(laspy.__version__)"
```

如果缺失，优先在服务器环境中安装：

```bash
conda activate fa-u
pip install laspy
```

基础转换命令：

```bash
cd /home/team-lu/harrison_workspace/projects/ForestAgent
conda activate fa-u

python scripts/convert_las_to_uni3d_npy.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/TLS \
  --output-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_batch_inputs \
  --pattern "*.las" \
  --limit 10 \
  --num-points 10000 \
  --seed 42
```

如果希望保留 LAS 中真实 RGB，且文件包含 `red/green/blue` 字段：

```bash
python scripts/convert_las_to_uni3d_npy.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/TLS \
  --output-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_batch_inputs_rgb \
  --pattern "*.las" \
  --limit 10 \
  --num-points 10000 \
  --seed 42 \
  --include-rgb
```

转换规则：

- 输出 `.npy` 为 `float32`。
- 无 RGB 或不加 `--include-rgb` 时输出 `[N, 3]`。
- 加 `--include-rgb` 且 LAS 有 `red/green/blue` 字段时输出 `[N, 6]`。
- RGB 会归一化到 `[0, 1]`。
- 点数大于 `--num-points` 时做 deterministic 随机采样。
- 点数小于 `--num-points` 时本版跳过并记录失败，不做重复采样。
- 所有成功和失败样本都会写入 `manifest.jsonl`。

后续 batch extraction 的 `--input-dir` 应该指向转换脚本的 `--output-dir`，例如：

```bash
--input-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_batch_inputs
```

## 5. 单样本 sanity check 命令模板

如果只想复查单棵树真实 forward：

```bash
cd /home/team-lu/harrison_workspace/projects/ForestAgent
conda activate fa-u
export CUDA_VISIBLE_DEVICES=1

python scripts/sanity_check_uni3d_extractor.py \
  --xyz-path /home/team-lu/harrison_workspace/data/forestagent/uni3d_inputs/single_tree_xyz_10000.npy \
  --checkpoint-path /home/team-lu/harrison_workspace/checkpoints/Uni3D/modelzoo/uni3d-b/model.pt \
  --uni3d-repo-path /home/team-lu/harrison_workspace/projects/Uni3D \
  --device cuda \
  --pc-model eva02_base_patch14_448 \
  --pc-feat-dim 768 \
  --embed-dim 1024 \
  --pc-encoder-dim 512 \
  --num-group 512 \
  --group-size 64
```

`pc-model`、`pc-feat-dim`、`embed-dim`、`pc-encoder-dim`、`num-group`、`group-size` 必须和 checkpoint scale 匹配。已验证的 `uni3d-b/model.pt` 使用上面的参数。

## 6. 小批量 batch extraction 命令模板

输入目录中建议先放 5-10 个 `.npy` 文件：

```bash
cd /home/team-lu/harrison_workspace/projects/ForestAgent
conda activate fa-u
export CUDA_VISIBLE_DEVICES=1

python scripts/batch_extract_uni3d_embeddings.py \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_batch_inputs \
  --output-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/sanity_001 \
  --checkpoint-path /home/team-lu/harrison_workspace/checkpoints/Uni3D/modelzoo/uni3d-b/model.pt \
  --uni3d-repo-path /home/team-lu/harrison_workspace/projects/Uni3D \
  --device cuda \
  --pc-model eva02_base_patch14_448 \
  --pc-feat-dim 768 \
  --embed-dim 1024 \
  --pc-encoder-dim 512 \
  --num-group 512 \
  --group-size 64 \
  --limit 10 \
  --pattern "*.npy"
```

输出内容：

- 每棵树一个 `.npz`，包含 `embedding_raw`、`embedding_l2`、`tree_id`、`source_path`、`metadata`。
- `summary.json`，记录成功数量、失败数量、每个样本的 shape、finite、L2 norm、输出路径和失败原因。

如果需要从 manifest 读取：

```bash
python scripts/batch_extract_uni3d_embeddings.py \
  --manifest-path /home/team-lu/harrison_workspace/data/forestagent/uni3d_batch_manifest.txt \
  --input-dir /home/team-lu/harrison_workspace/data/forestagent/uni3d_batch_inputs \
  --output-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/sanity_001 \
  --checkpoint-path /home/team-lu/harrison_workspace/checkpoints/Uni3D/modelzoo/uni3d-b/model.pt \
  --uni3d-repo-path /home/team-lu/harrison_workspace/projects/Uni3D \
  --device cuda \
  --pc-model eva02_base_patch14_448 \
  --pc-feat-dim 768 \
  --embed-dim 1024 \
  --pc-encoder-dim 512 \
  --num-group 512 \
  --group-size 64 \
  --limit 10
```

## 7. Similarity analysis 命令模板

```bash
cd /home/team-lu/harrison_workspace/projects/ForestAgent
conda activate fa-u

python scripts/analyze_uni3d_embedding_similarity.py \
  --embedding-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/sanity_001 \
  --output-dir /home/team-lu/harrison_workspace/outputs/forestagent/uni3d_embeddings/sanity_001_analysis \
  --top-k 5
```

输出内容：

- `cosine_similarity.csv`：两两 cosine similarity 矩阵。
- `nearest_neighbors.json`：每棵树 top-k 最近邻。
- `similarity_summary.json`：样本数、embedding dim、pairwise similarity 的 min/max/mean/std、重复/近重复检查、NaN/Inf 检查。

## 8. 如何判断初步通过

可以认为“小批量 sanity check 初步通过”的条件：

- 每棵树都成功生成 `.npz`。
- `summary.json` 中 `success_count` 等于候选样本数，`failure_count=0`。
- 每棵树 `embedding_raw` 和 `embedding_l2` 都 finite。
- `embedding_l2` 的 L2 norm 接近 1。
- 每棵树 metadata 完整，包含 `feature_shape`、`point_count`、`rgb_source`、`checkpoint_path`、`model_builder` 等字段。
- 相似度分析可以生成三个输出文件。
- 不同树之间 pairwise similarity 不应全部异常接近 1。
- 如果从 LAS 转换开始，`manifest.jsonl` 中成功样本的 `shape` 应为 `[10000, 3]` 或 `[10000, 6]`。

注意：这只能说明 extractor 可运行、embedding 有基本分布差异；还不能说明它对林木测量或报告质量有效。

## 9. 如何判断还不能通过

需要暂停并定位的情况：

- CUDA 不可用：`torch.cuda.is_available()` 为 `False`。
- `pointnet2_ops` import 失败。
- 官方 Uni3D `models.uni3d` import 失败。
- checkpoint 文件不存在或 top-level key 不符合预期。
- `state_dict` shape mismatch，通常说明 checkpoint scale 和命令行模型参数不匹配。
- 输入点云不是 `[N, 3]` 或 `[N, 6]`。
- LAS 转换阶段 `laspy` import 失败。
- LAS 点数少于 `--num-points`，转换脚本会跳过并记录失败。
- 期望 RGB 但 LAS 没有 `red/green/blue` 字段，此时输出会退回 `[N, 3]`，不会伪造 RGB。
- 点数太少，小于 `max(num_group, group_size)`。
- forward 报 CUDA 错误。
- embedding 中出现 NaN 或 Inf。
- 两次单样本重复提取不稳定。
- 所有小批量 embedding 几乎完全一样，pairwise similarity 全部接近 1。

## 10. 当前边界

- 本工具不使用 mock extractor。
- 本工具不接入 q1/q2/q3 baseline。
- 本工具不接入主问答系统。
- 本工具不下载 checkpoint。
- 本工具不安装依赖。
- LAS 转换脚本只负责 `.las -> .npy`，不运行真实 Uni3D forward。
- 当前 Codex 本地测试只能验证脚本逻辑；真实 Uni3D forward 必须由服务器 GPU 环境验证。
