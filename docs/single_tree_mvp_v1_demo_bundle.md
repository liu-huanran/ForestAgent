# Single-Tree MVP v1 Demo Bundle

## 1. 这是什么

这份 demo bundle 是给当前 **single-tree analysis MVP v1** 做现场演示用的固定样例集。

范围只覆盖：

- `q1_dbh`
- `q2_height`
- `q3_crown_width`
- `tree_report_q123`

不包含：

- `q4-q8`
- agent 规划
- 真实 LLM 推理
- 倾斜 / 质量 / 形态 / 风险判断

## 2. 为什么先做 demo bundle

当前项目已经具备：

- q1/q2/q3 冻结 baseline
- 单木 CLI 入口
- `json_summary`
- `report_text`

所以最合适的下一步不是再扩工具，而是先把一套**可稳定复现的 demo 样例**固定下来，方便：

- 现场演示
- 后续接入说明
- 下一阶段 LLM 轻接入时做对照

## 3. 当前固定样本

这版先固定 3 个 TLS / ground 单木样本：

- `150_59`
- `150_24`
- `151_17`

对应点云路径：

- `data/TLS/150/150_59.las`
- `data/TLS/150/150_24.las`
- `data/TLS/151/151_17.las`

这些样本是从当前 q1/q2/q3 已完成验证结果里挑出来的，目的是优先保证演示稳定性。

## 4. 固定演示命令类型

每个样本都固定跑下面 5 类请求：

1. 单独查胸径
2. 单独查树高
3. 单独查冠幅
4. 生成简短单木报告
5. 一个失败/不支持问题

失败问题固定为：

```text
这棵树有倒伏风险吗？
```

这样可以明确展示：

- 支持的意图
- 不支持的意图
- 成功输出
- 失败行为

## 5. 如何生成 demo bundle

在项目根目录执行：

```powershell
python -m forestagent.demo_build_mvp_bundle
```

如果想自己指定输出目录：

```powershell
python -m forestagent.demo_build_mvp_bundle --output-dir outputs/single_tree_mvp_v1_demo_bundle_custom
```

如果以后要显式指定 direct geometry 配置：

```powershell
python -m forestagent.demo_build_mvp_bundle --config-path configs/direct_geometry.yaml
```

## 6. 输出目录结构

生成后会得到一个 demo bundle 目录，里面包含：

- `manifest.json`
- `150_59/`
- `150_24/`
- `151_17/`

每个样本子目录下面会固定有：

- `q1_dbh.json`
- `q2_height.json`
- `q3_crown_width.json`
- `tree_report_q123.json`
- `unsupported_question.json`

这些 JSON 都是直接调用当前 `run_single_tree_analysis(...)` 得到的原始输出，没有额外包装。

## 7. 如何现场演示

建议顺序：

1. 先展示 `q1_dbh.json`
2. 再展示 `q2_height.json`
3. 再展示 `q3_crown_width.json`
4. 再展示 `tree_report_q123.json`
5. 最后展示 `unsupported_question.json`

这样可以完整展示：

- 单指标查询
- 简短单木报告
- 失败/不支持问题时的行为

## 8. 和下一阶段 LLM 轻接入的关系

这份 demo bundle 也是下一阶段最好的输入基线。

后续如果做 “LLM 轻接入”，建议保持：

- 意图解析仍然优先规则路由
- 工具调用仍然固定
- `json_summary` 仍然是唯一事实来源
- LLM 只负责更自然的表达

也就是说，LLM 不应该自己猜：

- 胸径 / 树高 / 冠幅数值
- 倾斜 / 风险 / 质量判断
- 树种、健康状态、生态结论

## 9. 当前边界

这版 demo bundle 只是为了把 MVP v1 正式冻结成“可演示状态”。

它不会：

- 改任何 q1/q2/q3 算法
- 改 schema
- 改 `fixed_pipeline`
- 改 `task_rules`
- 接真实 LLM

所以它是一个**演示冻结层**，不是功能扩展层。
