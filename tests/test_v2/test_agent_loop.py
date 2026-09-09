"""Tests for v2 agent_loop — core loop logic with mocked LLM."""

import json
import pytest
from unittest.mock import MagicMock, patch

from project.src.v2.agent_loop import (
    AgentResult, SecHarness, agent_loop,
    _single_shot, _tool_loop, _force_verdict,
    SYSTEM_PROMPT_WITH_TOOLS, SYSTEM_PROMPT_NO_TOOLS,
)
from project.src.v2.llm_engine import LLMResponse
from project.src.v2.harness.audit import AuditRecordV2
from project.src.v2.harness.permissions import PermissionsManager
from project.src.v2.tools.registry import ToolRegistry


def _make_mock_llm(responses: list[str]):
    """Create a mock LLM engine that returns a sequence of responses."""
    llm = MagicMock()
    llm.base_model_name = "test-model"
    llm.adapter_path = None
    call_count = [0]

    def gen(messages, max_new_tokens=256, temperature=0.1):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return LLMResponse(
            text=responses[idx], input_tokens=100,
            output_tokens=50, latency_ms=100.0,
        )

    llm.generate = gen
    return llm


class TestSingleShot:
    """E1/E2: No harness, single LLM call."""

    def test_direct_verdict(self):
        llm = _make_mock_llm(['{"verdict": "benign", "confidence": 0.9, "reasoning": "normal"}'])
        harness = SecHarness(llm=llm, harness_enabled=False)
        result = agent_loop("proto=tcp ; state=FIN", harness)
        assert result.verdict == "benign"
        assert result.confidence == 0.9
        assert result.termination == "direct_verdict"

    def test_classify_in_single_shot(self):
        llm = _make_mock_llm(['{"tool": "classify", "args": {"verdict": "attack", "attack_type": "DoS", "confidence": 0.85, "reasoning": "flood"}}'])
        harness = SecHarness(llm=llm, harness_enabled=False)
        result = agent_loop("proto=tcp ; state=CON ; sbytes=very_high", harness)
        assert result.verdict == "attack"
        assert result.termination == "classify"

    def test_parse_fail(self):
        llm = _make_mock_llm(["I don't know what to do"])
        harness = SecHarness(llm=llm, harness_enabled=False)
        result = agent_loop("proto=tcp", harness)
        assert result.verdict == "benign"
        assert result.termination == "parse_fail"

    def test_audit_populated(self):
        llm = _make_mock_llm(['{"verdict": "attack", "attack_type": "DoS", "confidence": 0.8}'])
        harness = SecHarness(llm=llm, harness_enabled=False)
        result = agent_loop("proto=tcp", harness, ground_truth="DoS")
        assert result.audit is not None
        assert result.audit.detection_correct is True
        assert result.audit.num_llm_calls == 1


class TestToolLoop:
    """E3/E4: Harness enabled with tool calls."""

    def test_direct_classify(self):
        """Agent immediately calls classify."""
        llm = _make_mock_llm([
            '{"tool": "classify", "args": {"verdict": "benign", "confidence": 0.9, "reasoning": "all normal"}}',
        ])
        harness = SecHarness(llm=llm, harness_enabled=True, permissions_enabled=False)
        result = agent_loop("proto=tcp ; state=FIN", harness)
        assert result.verdict == "benign"
        assert result.termination == "classify"

    def test_tool_then_classify(self):
        """Agent calls check_anomaly then classify."""
        responses = [
            '{"tool": "check_anomaly", "args": {"traffic_text": "proto=tcp"}}',
            '{"tool": "classify", "args": {"verdict": "attack", "attack_type": "DoS", "confidence": 0.9, "reasoning": "ML confirms"}}',
        ]
        llm = _make_mock_llm(responses)
        # Mock check_anomaly
        harness = SecHarness(llm=llm, harness_enabled=True, permissions_enabled=False)
        harness.registry.register("check_anomaly", lambda traffic_text="": json.dumps({
            "prediction": "DoS", "confidence": 0.92, "top3": "DoS=0.92 | Normal=0.05",
        }))
        result = agent_loop("proto=tcp ; state=CON ; sbytes=very_high", harness)
        assert result.verdict == "attack"
        assert result.audit.total_steps == 2  # check_anomaly + classify both recorded

    def test_max_steps_fallback(self):
        """Agent never classifies — falls back after max_steps."""
        responses = [
            '{"tool": "log_decision", "args": {"observation": "hmm", "implication": "maybe"}}',
        ] * 5  # Fill all steps with non-terminal tool calls
        llm = _make_mock_llm(responses)
        harness = SecHarness(llm=llm, harness_enabled=True, permissions_enabled=False, max_steps=3)
        harness.registry.register("log_decision", lambda **kw: '{"logged": true}')
        result = agent_loop("proto=tcp", harness)
        assert result.termination == "max_steps"
        assert result.verdict == "benign"  # No ML prediction, default benign

    def test_parse_fail_in_loop(self):
        """Agent outputs unparseable text."""
        llm = _make_mock_llm(["This is not valid JSON at all"])
        harness = SecHarness(llm=llm, harness_enabled=True, permissions_enabled=False)
        result = agent_loop("proto=tcp", harness)
        assert result.termination == "parse_fail"

    def test_escalate(self):
        llm = _make_mock_llm([
            '{"tool": "escalate", "args": {"reason": "uncertain", "partial_verdict": "attack", "confidence": 0.4}}',
        ])
        harness = SecHarness(llm=llm, harness_enabled=True, permissions_enabled=False)
        harness.registry.register("escalate", lambda **kw: json.dumps({"escalated": True}))
        result = agent_loop("proto=tcp", harness)
        assert result.termination == "escalate"
        assert result.audit.escalated is True


class TestSecHarness:
    """SecHarness configuration and prompt building."""

    def test_no_harness_prompt(self):
        llm = MagicMock()
        llm.base_model_name = "test"
        llm.adapter_path = None
        h = SecHarness(llm=llm, harness_enabled=False)
        prompt = h.build_system_prompt()
        assert "tool" not in prompt.lower() or "tools" not in prompt.lower()
        assert "behavioral analysis" in prompt.lower()

    def test_harness_prompt_has_tools(self):
        llm = MagicMock()
        llm.base_model_name = "test"
        llm.adapter_path = None
        h = SecHarness(llm=llm, harness_enabled=True)
        prompt = h.build_system_prompt()
        assert "classify" in prompt

    def test_observation_preprocessing(self):
        llm = MagicMock()
        llm.base_model_name = "test"
        llm.adapter_path = None
        h = SecHarness(llm=llm, harness_enabled=True)
        user_prompt = h.build_user_prompt("proto=tcp ; state=FIN ; sbytes=very_high ; dbytes=zero")
        assert "Pre-analysis" in user_prompt
        assert "anomalous" in user_prompt.lower()

    def test_no_observation_without_harness(self):
        llm = MagicMock()
        llm.base_model_name = "test"
        llm.adapter_path = None
        h = SecHarness(llm=llm, harness_enabled=False)
        user_prompt = h.build_user_prompt("proto=tcp ; state=FIN")
        assert "Pre-analysis" not in user_prompt


class TestForceVerdict:
    def test_with_ml(self):
        audit = AuditRecordV2()
        ml = {"prediction": "DoS", "confidence": 0.9}
        harness = MagicMock()
        result = _force_verdict(audit, ml, harness)
        assert result.verdict == "attack"
        assert result.termination == "max_steps"

    def test_without_ml(self):
        audit = AuditRecordV2()
        harness = MagicMock()
        result = _force_verdict(audit, None, harness)
        assert result.verdict == "benign"
        assert result.termination == "max_steps"
