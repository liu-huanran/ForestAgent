"""Lightweight config loading utilities for ForestAgent YAML files."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Any


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a mapping-only YAML subset from disk."""

    resolved_path = Path(path).resolve()
    return _load_yaml_mapping_cached(str(resolved_path))


@lru_cache(maxsize=None)
def _load_yaml_mapping_cached(path: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    data = _parse_yaml(text)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level YAML mapping.")
    return data


def _parse_yaml(text: str) -> Any:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" in raw_line:
            raise ValueError("Tabs are not supported in the YAML subset parser.")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, stripped))

    if not lines:
        return {}

    parsed, next_index = _parse_block(lines, start_index=0, indent=0)
    if next_index != len(lines):
        raise ValueError("Unexpected trailing YAML content.")
    return parsed


def _parse_block(
    lines: list[tuple[int, str]], start_index: int, indent: int
) -> tuple[Any, int]:
    if start_index >= len(lines):
        return {}, start_index

    _, first_content = lines[start_index]
    if first_content.startswith("- "):
        return _parse_list_block(lines, start_index, indent)
    return _parse_mapping_block(lines, start_index, indent)


def _parse_mapping_block(
    lines: list[tuple[int, str]], start_index: int, indent: int
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    index = start_index

    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation at line: {content!r}")
        if content.startswith("- "):
            raise ValueError(f"Unexpected list item in mapping block: {content!r}")

        key, separator, raw_value = content.partition(":")
        if not separator:
            raise ValueError(f"Invalid YAML mapping entry: {content!r}")

        key = key.strip()
        if not key:
            raise ValueError("YAML keys cannot be empty.")
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key!r}")

        value_text = raw_value.strip()
        if value_text:
            mapping[key] = _parse_scalar(value_text)
            index += 1
            continue

        child_index = index + 1
        if child_index >= len(lines) or lines[child_index][0] <= indent:
            raise ValueError(f"Expected nested block for key {key!r}.")

        child, index = _parse_block(lines, start_index=child_index, indent=indent + 2)
        mapping[key] = child

    return mapping, index


def _parse_list_block(
    lines: list[tuple[int, str]], start_index: int, indent: int
) -> tuple[list[Any], int]:
    items: list[Any] = []
    index = start_index

    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation at line: {content!r}")
        if not content.startswith("- "):
            break

        value_text = content[2:].strip()
        if value_text:
            items.append(_parse_scalar(value_text))
            index += 1
            continue

        child_index = index + 1
        if child_index >= len(lines) or lines[child_index][0] <= indent:
            raise ValueError("Expected nested block for list item.")

        child, index = _parse_block(lines, start_index=child_index, indent=indent + 2)
        items.append(child)

    return items, index


def _parse_scalar(value_text: str) -> Any:
    lowered = value_text.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if (
        len(value_text) >= 2
        and value_text[0] in {'"', "'"}
        and value_text[-1] == value_text[0]
    ):
        return ast.literal_eval(value_text)

    try:
        if any(char in value_text for char in (".", "e", "E")):
            return float(value_text)
        return int(value_text)
    except ValueError:
        return value_text

