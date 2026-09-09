"""
tool_parser.py — Robust parser for LLM-generated tool calls.

Handles unreliable outputs from 3B models: truncated JSON, markdown wrapping,
missing quotes, etc.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A parsed tool invocation."""
    name: str
    args: dict[str, Any]
    raw_text: str = ""


@dataclass
class DirectVerdict:
    """LLM output that directly contains a verdict (no tool call)."""
    verdict: str
    attack_type: str | None = None
    confidence: float = 0.5
    reasoning: str = ""
    raw_text: str = ""


@dataclass
class ParseFailure:
    """Could not parse any meaningful output."""
    raw_text: str = ""
    error: str = ""


def parse_tool_call(text: str) -> ToolCall | DirectVerdict | ParseFailure:
    """Parse LLM output into a ToolCall, DirectVerdict, or ParseFailure.

    Parsing strategy:
    1. Strip markdown code fences if present
    2. Try to find {"tool": ..., "args": ...} pattern
    3. Try to find {"verdict": ...} pattern (direct verdict)
    4. Fallback: regex field extraction from broken JSON
    """
    if not text or not text.strip():
        return ParseFailure(raw_text=text, error="empty output")

    cleaned = _strip_markdown(text.strip())

    # Strategy 1: full JSON tool call
    tool_call = _try_parse_tool_json(cleaned)
    if tool_call is not None:
        return ToolCall(name=tool_call["tool"], args=tool_call.get("args", {}), raw_text=text)

    # Strategy 2: full JSON direct verdict
    verdict = _try_parse_verdict_json(cleaned)
    if verdict is not None:
        return verdict

    # Strategy 3: regex extraction from broken/truncated JSON
    tool_call = _try_regex_tool(cleaned)
    if tool_call is not None:
        return ToolCall(name=tool_call["tool"], args=tool_call.get("args", {}), raw_text=text)

    verdict = _try_regex_verdict(cleaned)
    if verdict is not None:
        return verdict

    return ParseFailure(raw_text=text, error="no parseable tool call or verdict")


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _try_parse_tool_json(text: str) -> dict | None:
    """Try to parse a complete {"tool": ..., "args": ...} JSON."""
    # Find all JSON-like blocks
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if "tool" in obj and isinstance(obj["tool"], str):
                if "args" not in obj:
                    obj["args"] = {}
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _try_parse_verdict_json(text: str) -> DirectVerdict | None:
    """Try to parse a {"verdict": ...} JSON."""
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if "verdict" in obj:
                v = obj["verdict"].lower().strip()
                if v in ("benign", "attack"):
                    return DirectVerdict(
                        verdict=v,
                        attack_type=obj.get("attack_type"),
                        confidence=float(obj.get("confidence", 0.5)),
                        reasoning=str(obj.get("reasoning", "")),
                        raw_text=text,
                    )
        except (json.JSONDecodeError, ValueError, AttributeError):
            continue
    return None


def _try_regex_tool(text: str) -> dict | None:
    """Regex-based extraction for truncated/malformed tool JSON."""
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
    if not m:
        return None
    tool_name = m.group(1)

    args = {}
    # Try to extract args block (with or without closing brace for truncated JSON)
    args_match = re.search(r'"args"\s*:\s*\{([^}]*)\}?', text, re.DOTALL)
    if args_match:
        args_text = args_match.group(1)
        # Parse individual key-value pairs
        for kv in re.finditer(r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"|[\d.]+|true|false|null)', args_text):
            key = kv.group(1)
            val = kv.group(2)
            try:
                args[key] = json.loads(val)
            except json.JSONDecodeError:
                args[key] = val.strip('"')

    return {"tool": tool_name, "args": args}


def _try_regex_verdict(text: str) -> DirectVerdict | None:
    """Regex-based extraction for truncated verdict JSON."""
    m = re.search(r'"verdict"\s*:\s*"(benign|attack)"', text, re.IGNORECASE)
    if not m:
        return None

    verdict = m.group(1).lower()
    attack_type = None
    confidence = 0.5
    reasoning = ""

    m2 = re.search(r'"attack_type"\s*:\s*"([^"]*)"', text)
    if m2:
        attack_type = m2.group(1)

    m3 = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    if m3:
        try:
            confidence = float(m3.group(1))
        except ValueError:
            pass

    m4 = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    if m4:
        reasoning = m4.group(1)

    return DirectVerdict(
        verdict=verdict,
        attack_type=attack_type,
        confidence=confidence,
        reasoning=reasoning,
        raw_text=text,
    )
