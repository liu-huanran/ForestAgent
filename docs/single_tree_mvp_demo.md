# Single-Tree Analysis MVP v1 Demo

## 1. 这版 MVP 是什么

这版 MVP 是一个**最小单木分析入口**，只基于当前已经冻结的 direct geometry baseline：

- `q1_dbh baseline v1.1`
- `q2_height baseline v1.1`
- `q3_crown_width baseline v1.1`

它只做固定工具编排：

- `estimate_dbh`
- `estimate_height`
- `estimate_crown_width`

然后输出两层结果：

- `json_summary`
- `report_text`

这版 MVP **不做**复杂 agent 规划，也**不接**真实 LLM SDK。

## 2. 支持哪些输入方式

### 方式 A：`--task-id`

适合明确知道要查什么指标的情况。

支持的 `task_id`：

- `q1_dbh`
- `q2_height`
- `q3_crown_width`
- `tree_report_q123`

### 方式 B：`--question`

适合现场直接用简单自然语言提问。

当前只支持**有限规则映射**，不是自由问答：

- 命中 `胸径 / dbh` -> `q1_dbh`
- 命中 `树高 / 高度 / height` -> `q2_height`
- 命中 `冠幅 / 冠宽 / crown width` -> `q3_crown_width`
- 命中 `报告 / 总结 / summary / report` -> `tree_report_q123`

如果问题同时命中多个指标，或者没有命中任何支持意图，会返回失败结果。

## 3. 最小运行命令示例

下面命令都假设你已经在项目根目录 `D:\Programming\LLM\ForestAgent` 下。

### 单独查胸径

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id q1_dbh
```

### 单独查树高

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.laz> --question "这棵树的树高是多少？"
```

### 单独查冠幅

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id q3_crown_width
```

### 生成简短单木报告

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --question "给我一个简短单木报告"
```

### 显式指定 direct geometry 配置

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id q2_height --config-path configs/direct_geometry.yaml
```

## 4. 输出说明

CLI 会直接打印一个 JSON，核心字段如下：

- `request`
  - 原始请求信息
  - 包括 `task_id`、`question`、`resolved_intent`
- `input`
  - 点云路径和 `format`
- `tool_results`
  - 本次实际执行过的工具结果
- `json_summary`
  - 结构化指标摘要
  - 固定包含：
    - `dbh_cm`
    - `height_m`
    - `crown_width_m`
- `report_text`
  - 简短文本报告
- `status`
  - `success` 或 `failed`
- `message`
  - 总体说明或失败原因

### `json_summary` 的行为

- 如果某项指标执行成功，会保留：
  - `value`
  - `unit`
  - `confidence`
  - `status`
  - `message`
- 如果该指标本次没有执行，对应字段就是 `null`
- 如果该指标执行失败，会保留失败状态和失败消息，不会编造数值

### `report_text` 的行为

- 单指标查询时，只返回一个短句
- `tree_report_q123` 时，返回一个 q1/q2/q3-only 的简短单木报告
- 如果失败，只返回失败模板，不会猜测未成功估计的值

## 5. 失败输入时会发生什么

### 例 1：未知任务

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id q9_unknown
```

预期行为：

- `status = "failed"`
- `tool_results = {}`
- `message` 提示当前支持的任务集合

### 例 2：歧义问题

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --question "这棵树的胸径和树高是多少？"
```

预期行为：

- `status = "failed"`
- 不执行工具
- 返回“无法唯一解析任务意图”的失败消息

### 例 3：工具失败

如果某项工具在运行中失败：

- 单指标请求：整体失败
- `tree_report_q123`：整体失败
- 但 `tool_results` 会保留已经执行过的工具结果，方便调试和演示

## 6. 可选 demo script

如果你想一次连续演示多个请求，可以运行：

```powershell
python -m forestagent.demo_single_tree_mvp --point-cloud <path-to-tree.las>
```

它会固定演示：

- `q1_dbh`
- `q2_height`
- `q3_crown_width`
- `tree_report_q123`

如果想顺手演示失败输入：

```powershell
python -m forestagent.demo_single_tree_mvp --point-cloud <path-to-tree.las> --include-failure-demo
```

## 7. 这版 MVP 明确不支持什么

这版只支持 q1/q2/q3 和一个 q1/q2/q3-only 的简短报告。

明确**不支持**：

- `q4-q8`
- 倾斜判断
- 点云质量判断
- 形态判断
- 风险判断
- 自由 agent 规划
- 真实 LLM 推理
- 树种、健康状态、生态含义等自由扩展结论

## 8. 现场演示建议

建议准备一个单木 `.las` 或 `.laz` 点云，然后按下面顺序演示：

1. `q1_dbh`
2. `q2_height`
3. `q3_crown_width`
4. `tree_report_q123`
5. 一个失败输入示例

这样可以一次性展示：

- 指标查询
- 结构化输出
- 模板报告
- 失败行为

而且整个演示不依赖 Excel、`DataCatalog` 或额外评估流程。
