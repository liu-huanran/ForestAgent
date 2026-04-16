"""Tests for the optional local Ollama verbalizer."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from forestagent.verbalizers.ollama_verbalizer import (
    OllamaVerbalizerError,
    build_ollama_verbalizer_messages,
    verbalize_json_summary_with_ollama,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class OllamaVerbalizerTests(unittest.TestCase):
    def test_build_prompt_uses_json_summary_only(self) -> None:
        summary = {
            "dbh_cm": {"value": 11.5, "unit": "cm", "status": "success", "message": "ok"},
            "height_m": None,
            "crown_width_m": None,
        }

        messages = build_ollama_verbalizer_messages(summary)

        self.assertEqual(len(messages), 2)
        self.assertIn("json_summary", messages[1]["content"])
        self.assertIn("11.5", messages[1]["content"])
        self.assertNotIn("synthetic_tree.las", messages[1]["content"])
        self.assertIn("禁止补全缺失值", messages[0]["content"])
        self.assertIn("禁止提及准确、可靠、健康、风险", messages[0]["content"])

    def test_verbalizer_returns_text_on_success(self) -> None:
        with patch(
            "forestagent.verbalizers.ollama_verbalizer.urlopen",
            return_value=_FakeResponse({"message": {"content": "这棵树胸径约为 11.50 cm。"}}),
        ):
            text = verbalize_json_summary_with_ollama(
                {"dbh_cm": {"value": 11.5, "unit": "cm", "status": "success", "message": "ok"}}
            )

        self.assertEqual(text, "这棵树胸径约为 11.50 cm。")

    def test_verbalizer_raises_on_invalid_response(self) -> None:
        with patch(
            "forestagent.verbalizers.ollama_verbalizer.urlopen",
            return_value=_FakeResponse({"message": {}}),
        ):
            with self.assertRaises(OllamaVerbalizerError):
                verbalize_json_summary_with_ollama(
                    {"dbh_cm": {"value": 11.5, "unit": "cm", "status": "success", "message": "ok"}}
                )
