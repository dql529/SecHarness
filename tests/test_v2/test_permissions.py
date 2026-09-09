"""Tests for v2 permissions — enhanced validation, confidence gate, consistency checker."""

import pytest
from project.src.v2.harness.permissions import (
    FormatValidator,
    ConfidenceGate,
    ConsistencyChecker,
    OutputLimiter,
    PermissionsManager,
    PermissionResult,
)


class TestFormatValidator:
    def setup_method(self):
        self.v = FormatValidator()

    def test_valid_args(self):
        r = self.v.check({"verdict": "attack", "attack_type": "DoS", "confidence": 0.9})
        assert r.allowed
        assert not r.violations

    def test_invalid_verdict(self):
        r = self.v.check({"verdict": "maybe", "confidence": 0.5})
        assert r.allowed  # still allowed, but modified
        assert r.modified_args["verdict"] == "benign"
        assert any("invalid verdict" in v for v in r.violations)

    def test_confidence_out_of_range(self):
        r = self.v.check({"verdict": "benign", "confidence": 1.5})
        assert r.modified_args["confidence"] == 1.0

    def test_attack_without_type(self):
        r = self.v.check({"verdict": "attack", "confidence": 0.8})
        assert any("attack_type required" in v for v in r.violations)

    def test_benign_no_type_ok(self):
        r = self.v.check({"verdict": "benign", "confidence": 0.8})
        assert not r.violations


class TestConfidenceGate:
    def setup_method(self):
        self.gate = ConfidenceGate(threshold=0.3)
        self.ml = {"prediction": "DoS", "confidence": 0.92}

    def test_above_threshold(self):
        r = self.gate.check({"verdict": "attack", "confidence": 0.8}, self.ml)
        assert r.allowed
        assert r.modified_args is None

    def test_below_threshold_fallback(self):
        r = self.gate.check({"verdict": "benign", "confidence": 0.2}, self.ml)
        assert r.allowed
        assert r.modified_args["verdict"] == "attack"
        assert r.modified_args["attack_type"] == "DoS"
        assert r.modified_args["confidence"] == 0.92

    def test_below_threshold_no_ml(self):
        r = self.gate.check({"verdict": "benign", "confidence": 0.1}, None)
        assert r.allowed
        assert r.modified_args is None  # No ML to fall back to

    def test_below_threshold_ml_benign(self):
        ml = {"prediction": "Normal", "confidence": 0.85}
        r = self.gate.check({"verdict": "attack", "confidence": 0.2}, ml)
        assert r.modified_args["verdict"] == "benign"


class TestConsistencyChecker:
    def setup_method(self):
        self.checker = ConsistencyChecker(ml_confidence_threshold=0.9)

    def test_agreement(self):
        ml = {"prediction": "DoS", "confidence": 0.95}
        r = self.checker.check({"verdict": "attack"}, ml)
        assert r.allowed
        assert not r.needs_retry

    def test_disagreement_high_ml(self):
        ml = {"prediction": "DoS", "confidence": 0.95}
        r = self.checker.check({"verdict": "benign"}, ml)
        assert not r.allowed
        assert r.needs_retry
        assert "disagrees" in r.retry_message

    def test_disagreement_low_ml(self):
        ml = {"prediction": "DoS", "confidence": 0.6}
        r = self.checker.check({"verdict": "benign"}, ml)
        assert r.allowed  # ML not confident enough to trigger

    def test_no_ml(self):
        r = self.checker.check({"verdict": "benign"}, None)
        assert r.allowed


class TestOutputLimiter:
    def test_short_reasoning(self):
        limiter = OutputLimiter(max_chars=500)
        r = limiter.check({"reasoning": "short text"})
        assert not r.violations

    def test_long_reasoning(self):
        limiter = OutputLimiter(max_chars=50)
        r = limiter.check({"reasoning": "x" * 100})
        assert r.modified_args is not None
        assert len(r.modified_args["reasoning"]) == 53  # 50 + "..."


class TestPermissionsManager:
    def test_disabled(self):
        pm = PermissionsManager(enabled=False)
        r = pm.check_classify({"verdict": "invalid"})
        assert r.allowed
        assert r.modified_args is None

    def test_full_pipeline_valid(self):
        pm = PermissionsManager(enabled=True)
        r = pm.check_classify(
            {"verdict": "attack", "attack_type": "DoS", "confidence": 0.9, "reasoning": "ok"},
            ml_prediction={"prediction": "DoS", "confidence": 0.9},
        )
        assert r.allowed

    def test_consistency_retry(self):
        pm = PermissionsManager(enabled=True, ml_conflict_threshold=0.9)
        ml = {"prediction": "DoS", "confidence": 0.95}
        r = pm.check_classify({"verdict": "benign", "confidence": 0.7, "reasoning": "looks ok"}, ml)
        assert r.needs_retry

    def test_consistency_retry_then_accept(self):
        pm = PermissionsManager(enabled=True, ml_conflict_threshold=0.9)
        ml = {"prediction": "DoS", "confidence": 0.95}
        # Second attempt — already_retried=True, so consistency check skipped
        r = pm.check_classify(
            {"verdict": "benign", "confidence": 0.7, "reasoning": "still benign"},
            ml, already_retried=True,
        )
        assert r.allowed
        assert not r.needs_retry

    def test_low_confidence_fallback(self):
        pm = PermissionsManager(enabled=True, confidence_threshold=0.3)
        ml = {"prediction": "Exploits", "confidence": 0.88}
        r = pm.check_classify(
            {"verdict": "benign", "confidence": 0.1, "reasoning": "unsure"},
            ml, already_retried=True,
        )
        assert r.allowed
        assert r.modified_args["verdict"] == "attack"
        assert r.modified_args["attack_type"] == "Exploits"
