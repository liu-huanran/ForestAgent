"""Minimal q1/q2/q3-only report renderer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def render_minimal_tree_report(
    *,
    resolved_intent: str | None,
    json_summary: Mapping[str, Any],
    status: str,
    message: str,
) -> str:
    """Render a short q1/q2/q3-only report from structured summary data."""

    if status != "success":
        return f"本次单木分析未完成。原因：{message}。"

    dbh = json_summary.get("dbh_cm")
    height = json_summary.get("height_m")
    crown_width = json_summary.get("crown_width_m")

    if resolved_intent == "q1_dbh" and isinstance(dbh, Mapping):
        return f"该单木估计胸径为 {float(dbh['value']):.2f} {dbh['unit']}。"
    if resolved_intent == "q2_height" and isinstance(height, Mapping):
        return f"该单木估计树高为 {float(height['value']):.2f} {height['unit']}。"
    if resolved_intent == "q3_crown_width" and isinstance(crown_width, Mapping):
        return f"该单木估计冠幅为 {float(crown_width['value']):.2f} {crown_width['unit']}。"
    if (
        resolved_intent == "tree_report_q123"
        and isinstance(dbh, Mapping)
        and isinstance(height, Mapping)
        and isinstance(crown_width, Mapping)
    ):
        return (
            f"该单木估计胸径为 {float(dbh['value']):.2f} {dbh['unit']}，"
            f"树高为 {float(height['value']):.2f} {height['unit']}，"
            f"冠幅为 {float(crown_width['value']):.2f} {crown_width['unit']}。"
            "以上结果均来自 direct geometry baseline；本版未进行倾斜、质量、形态或风险判断。"
        )

    return f"本次单木分析未完成。原因：{message or '结构化结果不完整。'}。"
