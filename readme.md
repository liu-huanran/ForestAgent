# ForestAgent

ForestAgent 当前被冻结为一个 **pre-feature / pre-LLM reasoning** 的科研原型 baseline。

当前官方 baseline 是 **Baseline V0**：

- 面向单木点云
- 覆盖 `q1_dbh / q2_height / q3_crown_width / tree_report_q123`
- 使用固定流程与确定性 direct geometry backend
- 支持一个**可选**的本地 Ollama verbalizer，用于把已有 `json_summary` 改写成自然语言报告

注意：

- Ollama 在 Baseline V0 中**不是**决策器
- 它**不会**参与任务选择、工具选择、数值估计或规则判断
- 它只会在显式开启 `--use-local-llm-report` 时，对已经算出的结构化结果做文本改写

## 官方入口

推荐从这里开始：

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id tree_report_q123
```

如果你想显式启用本地 Ollama verbalizer：

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id tree_report_q123 --use-local-llm-report --ollama-model qwen3:1.7b
```

## 快速说明

当前 Baseline V0 主链路是：

1. `forestagent.cli`
2. `analyze-tree`
3. `forestagent.mvp.single_tree_analysis`
4. `task_id / question` 规则映射
5. `forestagent.backends.direct_geometry_backend`
6. `estimate_dbh / estimate_height / estimate_crown_width`
7. 模板报告
8. 可选本地 Ollama verbalizer

## 关键文档

- Baseline 定义与边界：`docs/baseline_v0.md`
- Baseline 成功样例：`docs/examples/baseline_v0_success.json`

## 最小验证

```powershell
python -m unittest tests.test_baseline_v0_smoke
```

这条测试会验证：

- 官方 CLI 主入口可跑通
- q1/q2/q3-only 报告主路径未损坏
- Baseline V0 的可选 Ollama verbalizer 路径仍然存在

## 当前明确不做

- q4 / q5 作为本次 baseline 冻结目标
- Uni3D 特征
- learned feature
- agent planning
- 用 LLM 直接做数值估计或任务决策

如果你要看最完整的 baseline 说明，请直接读 `docs/baseline_v0.md`。
