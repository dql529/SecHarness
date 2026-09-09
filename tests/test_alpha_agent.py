"""Unit tests for Agent-Alpha (behavioral analysis agent)."""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.agents.base_agent import BaseAgent, AgentVerdict
from src.agents.alpha_agent import AlphaAgent, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------
SAMPLE_KV = (
    "proto=tcp ; state=FIN ; service=http ; dur=medium ; spkts=high ; "
    "dpkts=low ; sbytes=very_high ; dbytes=low ; sload=high ; dload=low ; "
    "tcprtt=low ; synack=low ; ackdat=low ; ct_srv_src=medium ; ct_state_ttl=high"
)

SAMPLE_KV_BENIGN = (
    "proto=tcp ; state=FIN ; service=dns ; dur=low ; spkts=low ; "
    "dpkts=low ; sbytes=low ; dbytes=low ; sload=low ; dload=low ; "
    "tcprtt=low ; synack=low ; ackdat=low ; ct_srv_src=low ; ct_state_ttl=medium"
)


def _mock_ollama_response(verdict="attack", attack_type="DoS", confidence=0.85, reasoning="test"):
    """Build a fake Ollama API response."""
    return {
        "message": {
            "content": json.dumps({
                "verdict": verdict,
                "attack_type": attack_type,
                "confidence": confidence,
                "reasoning": reasoning,
            })
        },
        "eval_count": 50,
        "prompt_eval_count": 200,
    }


# ---------------------------------------------------------------------------
# Tests — AgentVerdict
# ---------------------------------------------------------------------------
class TestAgentVerdict:
    def test_to_dict(self):
        v = AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.9, reasoning="high sbytes")
        d = v.to_dict()
        assert d["verdict"] == "attack"
        assert d["attack_type"] == "DoS"
        assert d["confidence"] == 0.9

    def test_from_dict(self):
        d = {"verdict": "benign", "confidence": 0.7, "reasoning": "normal", "extra_key": 123}
        v = AgentVerdict.from_dict(d)
        assert v.verdict == "benign"
        assert v.confidence == 0.7

    def test_defaults(self):
        v = AgentVerdict(verdict="benign")
        assert v.attack_type is None
        assert v.confidence == 0.5
        assert v.reasoning == ""


# ---------------------------------------------------------------------------
# Tests — AlphaAgent
# ---------------------------------------------------------------------------
class TestAlphaAgent:
    def test_is_base_agent(self):
        agent = AlphaAgent(model="test")
        assert isinstance(agent, BaseAgent)
        assert agent.name == "Agent-Alpha"

    def test_repr(self):
        agent = AlphaAgent(model="llama3:8b")
        assert "llama3:8b" in repr(agent)

    @patch("src.agents.alpha_agent.requests.post")
    def test_analyze_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_ollama_response()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        agent = AlphaAgent(model="test")
        result = agent.analyze(SAMPLE_KV)

        assert result.verdict == "attack"
        assert result.attack_type == "DoS"
        assert result.confidence == 0.85
        assert result.agent_name == "Agent-Alpha"
        assert result.latency_ms > 0

    @patch("src.agents.alpha_agent.requests.post")
    def test_analyze_benign(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_ollama_response(
            verdict="benign", attack_type=None, confidence=0.92, reasoning="normal dns"
        )
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        agent = AlphaAgent(model="test")
        result = agent.analyze(SAMPLE_KV_BENIGN)

        assert result.verdict == "benign"
        assert result.attack_type is None
        assert result.confidence == 0.92

    @patch("src.agents.alpha_agent.requests.post")
    def test_analyze_malformed_json(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "I think this is an attack"}}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        agent = AlphaAgent(model="test")
        result = agent.analyze(SAMPLE_KV)

        # Should fallback to benign with 0 confidence
        assert result.verdict == "benign"
        assert result.confidence == 0.0
        assert "PARSE_FAIL" in result.reasoning

    @patch("src.agents.alpha_agent.requests.post")
    def test_analyze_with_markdown_fence(self, mock_post):
        """LLM sometimes wraps JSON in ```json ... ```."""
        inner = json.dumps({"verdict": "attack", "attack_type": "Recon", "confidence": 0.7, "reasoning": "scanning"})
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": f"```json\n{inner}\n```"}, "eval_count": 30}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        agent = AlphaAgent(model="test")
        result = agent.analyze(SAMPLE_KV)
        assert result.verdict == "attack"
        assert result.attack_type == "Recon"

    @patch("src.agents.alpha_agent.requests.post")
    def test_analyze_connection_error(self, mock_post):
        import requests as req
        mock_post.side_effect = req.ConnectionError("refused")

        agent = AlphaAgent(model="test")
        result = agent.analyze(SAMPLE_KV)

        assert result.verdict == "benign"
        assert result.confidence == 0.0
        assert "ERROR" in result.reasoning

    @patch("src.agents.alpha_agent.requests.post")
    def test_analyze_batch(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_ollama_response()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        agent = AlphaAgent(model="test", max_workers=2)
        texts = [SAMPLE_KV, SAMPLE_KV_BENIGN, SAMPLE_KV]
        results = agent.analyze_batch(texts)

        assert len(results) == 3
        assert all(r.agent_name == "Agent-Alpha" for r in results)
        assert all(r.latency_ms > 0 for r in results)

    @patch("src.agents.alpha_agent.requests.get")
    def test_health_check_ok(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        agent = AlphaAgent(model="llama3:8b")
        assert agent.check_health() is True

    @patch("src.agents.alpha_agent.requests.get")
    def test_health_check_model_missing(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "qwen2:7b"}]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        agent = AlphaAgent(model="llama3:8b")
        assert agent.check_health() is False

    def test_system_prompt_content(self):
        assert "behavioral analysis" in SYSTEM_PROMPT.lower()
        assert "JSON" in SYSTEM_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
