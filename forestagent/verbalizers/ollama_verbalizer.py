"""Minimal local Ollama verbalizer for q1/q2/q3 single-tree summaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaVerbalizerError(RuntimeError):
    """Raised when the local Ollama verbalizer cannot produce a report."""


def verbalize_json_summary_with_ollama(
    json_summary: Mapping[str, Any],
    *,
    model: str = "qwen3:1.7b",
    base_url: str = "http://localhost:11434/api",
    timeout_seconds: float = 60.0,
) -> str:
    """Generate a short Chinese report from structured q1/q2/q3 summary data."""

    endpoint = f"{base_url.rstrip('/')}/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": build_ollama_verbalizer_messages(json_summary),
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:  # pragma: no cover
        raise OllamaVerbalizerError(f"Ollama HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise OllamaVerbalizerError(f"Ollama connection failed: {exc.reason}") from exc
    except TimeoutError as exc:  # pragma: no cover
        raise OllamaVerbalizerError("Ollama request timed out.") from exc
    except json.JSONDecodeError as exc:
        raise OllamaVerbalizerError("Ollama returned invalid JSON.") from exc
    except Exception as exc:  # pragma: no cover
        raise OllamaVerbalizerError(f"Ollama verbalizer failed: {exc}") from exc

    message = response_payload.get("message")
    if not isinstance(message, Mapping):
        raise OllamaVerbalizerError("Ollama response missing message object.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OllamaVerbalizerError("Ollama response missing textual content.")
    return content.strip()


def build_ollama_verbalizer_messages(json_summary: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build a strict, summary-only prompt for the local Ollama verbalizer."""

    summary_text = json.dumps(json_summary, ensure_ascii=False, sort_keys=True, indent=2)
    system_prompt = (
        "你是 ForestAgent 的本地 verbalizer。"
        "你只能把给定 json_summary 中已经存在的数值、单位、状态和消息改写成 1 到 2 句自然中文。"
        "禁止补全缺失值。"
        "禁止新增任何事实、推测或判断。"
        "禁止提及准确、可靠、健康、风险、树种、形态、倾斜、质量等额外结论。"
        "如果某项为 null 或 failed，只能表述为未成功估计或未评估，不能猜测。"
        "输出只允许是自然中文报告正文，不要输出标题、列表、JSON 或解释。"
    )
    user_prompt = (
        "请仅根据下面的 json_summary 生成 1 到 2 句自然中文报告。\n"
        "json_summary:\n"
        f"{summary_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
