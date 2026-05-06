"""Markdown report generation for offline Uni3D experiments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from forestagent.experiments.uni3d_experiment_config import experiment_name


def generate_experiment_report(
    config: dict[str, Any],
    *,
    audit: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> str:
    """Generate a Chinese Markdown experiment record."""

    name = experiment_name(config)
    experiment = config.get("experiment", {})
    paths = config.get("paths", {})
    conversion = config.get("conversion", {})
    extractor = config.get("extractor", {})
    similarity = (config.get("analysis") or {}).get("similarity", {})
    probe = (config.get("analysis") or {}).get("probe", {})

    lines = [
        f"# 实验记录｜{name}",
        "",
        f"生成时间：`{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## 1. 实验目标",
        "",
        experiment.get("description", "未填写实验目标。"),
        "",
        "## 2. 实验版本",
        "",
        f"- 版本：`{experiment.get('version')}`",
        f"- 阶段：`{experiment.get('stage')}`",
        f"- 创建者：`{experiment.get('created_by', 'unknown')}`",
        f"- 备注：{experiment.get('notes', '') or '无'}",
        "",
        "## 3. 输入数据",
        "",
        f"- 原始 LAS：`{paths.get('raw_las_dir', '未配置')}`",
        f"- Uni3D NPY 输入：`{paths.get('npy_input_dir')}`",
        f"- 标签表：`{paths.get('label_path')}`",
        "",
        "## 4. 转换设置",
        "",
        f"- conversion.enabled：`{conversion.get('enabled')}`",
        f"- output_mode：`{conversion.get('output_mode', '未配置')}`",
        f"- color_policy：`{conversion.get('color_policy', '未配置')}`",
        f"- semantic_color_likely：`{conversion.get('semantic_color_likely', False)}`",
        f"- natural_rgb：`{conversion.get('natural_rgb', False)}`",
        "",
        "## 5. Uni3D extractor 设置",
        "",
        f"- checkpoint：`{paths.get('checkpoint_path')}`",
        f"- Uni3D repo：`{paths.get('uni3d_repo_path')}`",
        f"- pc_model：`{extractor.get('pc_model')}`",
        f"- embed_dim：`{extractor.get('embed_dim')}`",
        f"- num_group / group_size：`{extractor.get('num_group')}` / `{extractor.get('group_size')}`",
        "",
        "## 6. Embedding 结果",
        "",
        _audit_section_text(audit, "extraction"),
        "",
        "## 7. Similarity 结果",
        "",
        _audit_section_text(audit, "similarity"),
        "",
        "## 8. Probe 结果",
        "",
        _audit_section_text(audit, "probe"),
        "",
        "## 9. 异常 / 近重复 / RGB / site 观察",
        "",
        f"- similarity.top_k：`{similarity.get('top_k', 5)}`",
        f"- probe.splits：`{probe.get('splits', [])}`",
        f"- probe.rgb_source：`{probe.get('rgb_source', 'all')}`",
        "- 若 color_policy 使用 semantic/site/species 相关颜色，必须优先检查 site/RGB/label 泄漏风险。",
        "",
        "## 10. 当前结论",
        "",
        _current_conclusion(config, audit, comparison),
        "",
        "## 11. 不能下的结论",
        "",
        "- 不能说 Uni3D 已经接入主系统。",
        "- 不能说 Uni3D 已经替代 q1/q2/q3 几何工具。",
        "- 不能把 strict probe 或 similarity 的单轮结果直接写成论文最终结论。",
        "- 不能把 semantic/discrete color 描述成 natural RGB，除非原始 LAS 语义已被独立确认。",
        "",
        "## 12. 下一步",
        "",
        "- 在服务器上手动执行命令计划，完成缺失阶段。",
        "- 审计 conversion / extraction / similarity / probe 输出是否完整。",
        "- 对 mixed、xyz-only、color-all 做横向对比，优先看 leave-one-site-out 而不是 random split。",
        "",
    ]
    return "\n".join(lines)


def generate_memory_snippet(
    config: dict[str, Any],
    *,
    audit: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> str:
    """Generate a short PROJECT_MEMORY.md snippet."""

    name = experiment_name(config)
    lines = [
        f"- Uni3D offline experiment framework now tracks `{name}` via config-driven command planning, audit, comparison, and report generation.",
        "- This framework is offline-only: it does not run Uni3D forward, train Uni3D, or modify q1/q2/q3/main QA paths.",
    ]
    if audit:
        lines.append(f"- Latest audit status for `{name}`: `{audit.get('overall_status', 'unknown')}`.")
    if comparison:
        lines.append(f"- Multi-experiment comparison artifact count: `{comparison.get('experiment_count', 0)}` configs compared.")
    lines.append("- Generated reports remain engineering/probe records, not final paper conclusions.")
    return "\n".join(lines)


def _audit_section_text(audit: dict[str, Any] | None, section: str) -> str:
    if not audit:
        return "未运行 / 未验证。"
    check = (audit.get("checks") or {}).get(section)
    if not check:
        return "未运行 / 未验证。"
    return f"- 状态：`{check.get('status')}`\n- 说明：{check.get('reason', '')}\n- 路径：`{check.get('path', '')}`"


def _current_conclusion(
    config: dict[str, Any],
    audit: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
) -> str:
    status = audit.get("overall_status") if audit else "未审计"
    warnings = config.get("metadata", {}).get("warning") or ""
    extra = f" 配置警告：{warnings}" if warnings else ""
    compared = f" 已纳入 `{comparison.get('experiment_count')}` 个实验的对比。" if comparison else ""
    return (
        f"当前 `{experiment_name(config)}` 的实验状态为 `{config.get('experiment', {}).get('stage')}`，"
        f"审计状态为 `{status}`。{extra}{compared}"
    )

