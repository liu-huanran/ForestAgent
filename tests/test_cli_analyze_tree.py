"""Tests for analyze-tree CLI argument routing."""

from __future__ import annotations

import unittest

from forestagent.cli import build_parser


class AnalyzeTreeCliTests(unittest.TestCase):
    def test_analyze_tree_accepts_local_llm_args(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "analyze-tree",
                "--point-cloud",
                "demo_tree.las",
                "--question",
                "给我一个简短单木报告",
                "--use-local-llm-report",
                "--ollama-model",
                "qwen3:1.7b",
                "--ollama-timeout-seconds",
                "120",
            ]
        )

        self.assertEqual(args.command, "analyze-tree")
        self.assertTrue(args.use_local_llm_report)
        self.assertEqual(args.ollama_model, "qwen3:1.7b")
        self.assertEqual(args.ollama_timeout_seconds, 120.0)

    def test_analyze_tree_local_llm_timeout_has_safe_default(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "analyze-tree",
                "--point-cloud",
                "demo_tree.las",
                "--question",
                "给我一个简短单木报告",
            ]
        )

        self.assertEqual(args.ollama_timeout_seconds, 60.0)
