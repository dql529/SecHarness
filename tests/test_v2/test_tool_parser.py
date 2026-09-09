"""Tests for v2 tool_parser — robust JSON parsing from 3B model output."""

import pytest
from project.src.v2.tool_parser import (
    ToolCall, DirectVerdict, ParseFailure,
    parse_tool_call,
)


class TestNormalJSON:
    """Well-formed JSON outputs."""

    def test_tool_call(self):
        text = '{"tool": "check_anomaly", "args": {"traffic_text": "proto=tcp ; state=FIN"}}'
        result = parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.name == "check_anomaly"
        assert result.args["traffic_text"] == "proto=tcp ; state=FIN"

    def test_classify_call(self):
        text = '{"tool": "classify", "args": {"verdict": "attack", "attack_type": "DoS", "confidence": 0.92, "reasoning": "high sbytes"}}'
        result = parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.name == "classify"
        assert result.args["verdict"] == "attack"
        assert result.args["confidence"] == 0.92

    def test_direct_verdict(self):
        text = '{"verdict": "benign", "confidence": 0.85, "reasoning": "normal traffic"}'
        result = parse_tool_call(text)
        assert isinstance(result, DirectVerdict)
        assert result.verdict == "benign"
        assert result.confidence == 0.85

    def test_tool_no_args(self):
        text = '{"tool": "query_history"}'
        result = parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.name == "query_history"
        assert result.args == {}


class TestMarkdownWrapped:
    """JSON wrapped in markdown code fences."""

    def test_json_fence(self):
        text = '```json\n{"tool": "check_anomaly", "args": {"traffic_text": "proto=tcp"}}\n```'
        result = parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.name == "check_anomaly"

    def test_plain_fence(self):
        text = '```\n{"verdict": "attack", "attack_type": "DoS", "confidence": 0.9}\n```'
        result = parse_tool_call(text)
        assert isinstance(result, DirectVerdict)
        assert result.verdict == "attack"


class TestTruncatedJSON:
    """Truncated or malformed JSON."""

    def test_truncated_tool(self):
        text = '{"tool": "classify", "args": {"verdict": "attack", "attack_type": "DoS", "confidence": 0.8, "reasoning": "high sby'
        result = parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.name == "classify"
        assert result.args["verdict"] == "attack"
        assert result.args["confidence"] == 0.8

    def test_truncated_verdict(self):
        text = '{"verdict": "attack", "attack_type": "Exploits", "confidence": 0.75, "reasoning": "buffer over'
        result = parse_tool_call(text)
        assert isinstance(result, DirectVerdict)
        assert result.verdict == "attack"
        assert result.confidence == 0.75

    def test_extra_text_before(self):
        text = 'Based on my analysis, I will call:\n{"tool": "check_anomaly", "args": {"traffic_text": "proto=tcp"}}'
        result = parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.name == "check_anomaly"


class TestFailureCases:
    """Outputs that can't be parsed."""

    def test_empty(self):
        result = parse_tool_call("")
        assert isinstance(result, ParseFailure)

    def test_whitespace(self):
        result = parse_tool_call("   \n  ")
        assert isinstance(result, ParseFailure)

    def test_plain_text(self):
        result = parse_tool_call("I think this traffic looks normal based on the features.")
        assert isinstance(result, ParseFailure)

    def test_none_input(self):
        result = parse_tool_call(None)
        assert isinstance(result, ParseFailure)

    def test_invalid_verdict(self):
        text = '{"verdict": "maybe", "confidence": 0.5}'
        result = parse_tool_call(text)
        # Should be ParseFailure since "maybe" is not valid
        assert isinstance(result, ParseFailure)


class TestEdgeCases:
    """Edge cases for robustness."""

    def test_mixed_case_verdict(self):
        text = '{"verdict": "Attack", "attack_type": "DoS", "confidence": 0.8}'
        result = parse_tool_call(text)
        assert isinstance(result, DirectVerdict)
        assert result.verdict == "attack"

    def test_confidence_as_string(self):
        text = '{"tool": "classify", "args": {"verdict": "benign", "confidence": "0.7", "reasoning": "ok"}}'
        result = parse_tool_call(text)
        assert isinstance(result, ToolCall)
        # confidence might be string in args, that's OK — downstream handles conversion

    def test_nested_args(self):
        text = '{"tool": "classify", "args": {"verdict": "attack", "attack_type": "DoS", "confidence": 0.9, "reasoning": "high bytes"}}'
        result = parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.args["attack_type"] == "DoS"
