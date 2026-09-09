"""
test_consensus.py — Unit tests for Consensus Module and DetectionPipeline.
"""

import pytest

from project.src.agents.base_agent import AgentVerdict
from project.src.agents.consensus import ConsensusModule, ConsensusResult
from project.src.agents.pipeline import DetectionPipeline
from project.src.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Helpers: stub agents for pipeline tests
# ---------------------------------------------------------------------------
class StubAgent(BaseAgent):
    """Agent that returns a pre-configured verdict for testing."""

    def __init__(self, name: str, verdict: AgentVerdict):
        super().__init__(name)
        self._verdict = verdict

    def analyze(self, traffic_text: str) -> AgentVerdict:
        v = AgentVerdict(
            verdict=self._verdict.verdict,
            attack_type=self._verdict.attack_type,
            confidence=self._verdict.confidence,
            reasoning=self._verdict.reasoning,
            agent_name=self.name,
        )
        return v


# ---------------------------------------------------------------------------
# ConsensusModule tests
# ---------------------------------------------------------------------------
class TestConsensusAgreement:
    """Tests for agreement paths."""

    def setup_method(self):
        self.cm = ConsensusModule()

    def test_both_benign(self):
        a = AgentVerdict(verdict="benign", confidence=0.9, agent_name="Alpha")
        b = AgentVerdict(verdict="benign", confidence=0.8, agent_name="Beta")
        r = self.cm.resolve(a, b)
        assert r.final_verdict == "benign"
        assert r.consensus_type == "agreement"
        assert r.combined_confidence == pytest.approx(0.9 * 0.6 + 0.8 * 0.4)
        assert r.escalation_reason is None

    def test_both_attack_same_type(self):
        a = AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.95, agent_name="Alpha")
        b = AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.85, agent_name="Beta")
        r = self.cm.resolve(a, b)
        assert r.final_verdict == "attack"
        assert r.final_attack_type == "DoS"
        assert r.consensus_type == "agreement"

    def test_both_attack_one_no_type(self):
        a = AgentVerdict(verdict="attack", attack_type="Exploits", confidence=0.8, agent_name="Alpha")
        b = AgentVerdict(verdict="attack", attack_type=None, confidence=0.7, agent_name="Beta")
        r = self.cm.resolve(a, b)
        assert r.final_verdict == "attack"
        assert r.final_attack_type == "Exploits"
        assert r.consensus_type == "agreement"

    def test_both_attack_different_type(self):
        a = AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.9, agent_name="Alpha")
        b = AgentVerdict(verdict="attack", attack_type="Recon", confidence=0.8, agent_name="Beta")
        r = self.cm.resolve(a, b)
        assert r.final_verdict == "attack"
        assert r.consensus_type == "type_conflict"
        assert r.final_attack_type == "DoS"  # Alpha priority for type
        assert "conflict" in r.escalation_reason.lower()


class TestConsensusDisagreement:
    """Tests for disagreement paths."""

    def test_alpha_attack_beta_benign_escalate(self):
        cm = ConsensusModule(disagreement_strategy="escalate_all")
        a = AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.8, agent_name="Alpha")
        b = AgentVerdict(verdict="benign", confidence=0.7, agent_name="Beta")
        r = cm.resolve(a, b)
        assert r.final_verdict == "escalate"
        assert r.consensus_type == "alpha_only"
        assert r.final_attack_type == "DoS"

    def test_alpha_benign_beta_attack_escalate(self):
        cm = ConsensusModule(disagreement_strategy="escalate_all")
        a = AgentVerdict(verdict="benign", confidence=0.7, agent_name="Alpha")
        b = AgentVerdict(verdict="attack", attack_type="Recon", confidence=0.8, agent_name="Beta")
        r = cm.resolve(a, b)
        assert r.final_verdict == "escalate"
        assert r.consensus_type == "beta_only"
        assert r.final_attack_type == "Recon"

    def test_alpha_priority_strategy(self):
        cm = ConsensusModule(disagreement_strategy="alpha_priority")
        a = AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.8, agent_name="Alpha")
        b = AgentVerdict(verdict="benign", confidence=0.7, agent_name="Beta")
        r = cm.resolve(a, b)
        assert r.final_verdict == "attack"
        assert r.consensus_type == "alpha_only"

    def test_beta_priority_strategy(self):
        cm = ConsensusModule(disagreement_strategy="beta_priority")
        a = AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.8, agent_name="Alpha")
        b = AgentVerdict(verdict="benign", confidence=0.7, agent_name="Beta")
        r = cm.resolve(a, b)
        assert r.final_verdict == "benign"
        assert r.consensus_type == "alpha_only"


class TestConsensusUncertain:
    """Tests for low-confidence / uncertain path."""

    def test_both_low_confidence(self):
        cm = ConsensusModule(confidence_threshold=0.4)
        a = AgentVerdict(verdict="attack", confidence=0.2, agent_name="Alpha")
        b = AgentVerdict(verdict="benign", confidence=0.3, agent_name="Beta")
        r = cm.resolve(a, b)
        assert r.final_verdict == "escalate"
        assert r.consensus_type == "uncertain"
        assert "threshold" in r.escalation_reason.lower()

    def test_one_low_one_high_not_uncertain(self):
        cm = ConsensusModule(confidence_threshold=0.4)
        a = AgentVerdict(verdict="attack", confidence=0.9, agent_name="Alpha")
        b = AgentVerdict(verdict="benign", confidence=0.2, agent_name="Beta")
        r = cm.resolve(a, b)
        # Not uncertain because Alpha is above threshold
        assert r.consensus_type != "uncertain"


class TestConsensusConfig:
    """Tests for configuration and edge cases."""

    def test_custom_weights(self):
        cm = ConsensusModule(alpha_weight=0.7, beta_weight=0.3)
        a = AgentVerdict(verdict="benign", confidence=1.0, agent_name="Alpha")
        b = AgentVerdict(verdict="benign", confidence=0.0, agent_name="Beta")
        r = cm.resolve(a, b)
        assert r.combined_confidence == pytest.approx(0.7)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            ConsensusModule(alpha_weight=0.5, beta_weight=0.3)

    def test_is_disagreement_property(self):
        r = ConsensusResult(
            final_verdict="escalate", consensus_type="alpha_only",
            combined_confidence=0.5,
        )
        assert r.is_disagreement is True

        r2 = ConsensusResult(
            final_verdict="benign", consensus_type="agreement",
            combined_confidence=0.8,
        )
        assert r2.is_disagreement is False

    def test_to_dict(self):
        a = AgentVerdict(verdict="benign", confidence=0.9, agent_name="Alpha")
        b = AgentVerdict(verdict="benign", confidence=0.8, agent_name="Beta")
        r = ConsensusModule().resolve(a, b)
        d = r.to_dict()
        assert d["final_verdict"] == "benign"
        assert "alpha_verdict" in d

    def test_batch_resolve(self):
        cm = ConsensusModule()
        alphas = [
            AgentVerdict(verdict="benign", confidence=0.9, agent_name="Alpha"),
            AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.8, agent_name="Alpha"),
        ]
        betas = [
            AgentVerdict(verdict="benign", confidence=0.8, agent_name="Beta"),
            AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.7, agent_name="Beta"),
        ]
        results = cm.resolve_batch(alphas, betas)
        assert len(results) == 2
        assert results[0].final_verdict == "benign"
        assert results[1].final_verdict == "attack"


# ---------------------------------------------------------------------------
# DetectionPipeline tests
# ---------------------------------------------------------------------------
class TestPipeline:
    """Tests for the end-to-end detection pipeline."""

    def test_single_detect(self):
        alpha = StubAgent("Alpha", AgentVerdict(verdict="benign", confidence=0.9))
        beta = StubAgent("Beta", AgentVerdict(verdict="benign", confidence=0.8))
        pipe = DetectionPipeline(alpha, beta, parallel=False)
        r = pipe.detect("proto=tcp ; state=FIN ; dur=low")
        assert r.final_verdict == "benign"
        assert r.consensus_type == "agreement"

    def test_single_detect_parallel(self):
        alpha = StubAgent("Alpha", AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.9))
        beta = StubAgent("Beta", AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.8))
        pipe = DetectionPipeline(alpha, beta, parallel=True)
        r = pipe.detect("proto=tcp ; state=FIN ; dur=low")
        assert r.final_verdict == "attack"

    def test_batch_detect(self):
        alpha = StubAgent("Alpha", AgentVerdict(verdict="benign", confidence=0.9))
        beta = StubAgent("Beta", AgentVerdict(verdict="benign", confidence=0.8))
        pipe = DetectionPipeline(alpha, beta, parallel=False)
        texts = ["sample1", "sample2", "sample3"]
        results = pipe.detect_batch(texts, batch_size=2)
        assert len(results) == 3
        assert all(r.final_verdict == "benign" for r in results)

    def test_pipeline_with_custom_consensus(self):
        alpha = StubAgent("Alpha", AgentVerdict(verdict="attack", attack_type="DoS", confidence=0.8))
        beta = StubAgent("Beta", AgentVerdict(verdict="benign", confidence=0.7))
        cm = ConsensusModule(disagreement_strategy="alpha_priority")
        pipe = DetectionPipeline(alpha, beta, consensus=cm, parallel=False)
        r = pipe.detect("sample")
        assert r.final_verdict == "attack"

    def test_get_stats(self):
        alpha = StubAgent("Alpha", AgentVerdict(verdict="benign", confidence=0.9))
        beta = StubAgent("Beta", AgentVerdict(verdict="benign", confidence=0.8))
        pipe = DetectionPipeline(alpha, beta, parallel=False)
        results = pipe.detect_batch(["s1", "s2", "s3"])
        stats = pipe.get_stats(results)
        assert stats["total"] == 3
        assert stats["benign"] == 3
        assert stats["agreement_rate"] == 1.0

    def test_repr(self):
        alpha = StubAgent("Alpha", AgentVerdict(verdict="benign", confidence=0.9))
        beta = StubAgent("Beta", AgentVerdict(verdict="benign", confidence=0.8))
        pipe = DetectionPipeline(alpha, beta)
        assert "Alpha" in repr(pipe)
        assert "Beta" in repr(pipe)
