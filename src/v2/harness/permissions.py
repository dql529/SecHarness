"""
permissions.py — Enhanced Permissions for SecHarness v2.

Components:
- FormatValidator: verdict enum, confidence range, attack_type non-null
- ConfidenceGate: low-confidence fallback to ML prediction
- ConsistencyChecker: Agent vs ML disagreement → retry opportunity
- OutputLimiter: truncate reasoning to max_chars
- PermissionsManager: combines all four
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PermissionResult:
    """Result of a permissions check."""
    allowed: bool = True
    modified_args: Optional[dict[str, Any]] = None  # If args were adjusted
    needs_retry: bool = False  # If Agent should reconsider
    retry_message: str = ""  # Message to append to conversation
    violations: list[str] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class FormatValidator:
    """Validates classify tool arguments."""

    VALID_VERDICTS = {"benign", "attack"}

    def check(self, args: dict[str, Any]) -> PermissionResult:
        violations = []
        modified = dict(args)

        # Verdict enum
        verdict = str(args.get("verdict", "")).lower().strip()
        if verdict not in self.VALID_VERDICTS:
            violations.append(f"invalid verdict: {args.get('verdict')}")
            modified["verdict"] = "benign"

        # Confidence range
        try:
            conf = float(args.get("confidence", 0.5))
            clamped = max(0.0, min(1.0, conf))
            modified["confidence"] = clamped
            if clamped != conf:
                violations.append(f"confidence clamped: {conf} -> {clamped}")
        except (ValueError, TypeError):
            violations.append(f"invalid confidence: {args.get('confidence')}")
            modified["confidence"] = 0.5

        # attack_type required when verdict=attack
        if modified.get("verdict") == "attack" and not args.get("attack_type"):
            violations.append("attack_type required when verdict=attack")

        return PermissionResult(
            allowed=True,
            modified_args=modified if violations else None,
            violations=violations,
        )


class ConfidenceGate:
    """Falls back to ML prediction when Agent confidence is too low."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def check(
        self,
        args: dict[str, Any],
        ml_prediction: Optional[dict] = None,
    ) -> PermissionResult:
        try:
            conf = float(args.get("confidence", 0.5))
        except (ValueError, TypeError):
            conf = 0.5

        if conf < self.threshold and ml_prediction:
            ml_pred = ml_prediction.get("prediction", "Normal")
            ml_is_attack = ml_pred.lower() not in ("normal", "benign")
            fallback_verdict = "attack" if ml_is_attack else "benign"
            fallback_type = ml_pred if ml_is_attack else None
            ml_conf = ml_prediction.get("confidence", 0.5)

            logger.info(
                "ConfidenceGate: Agent confidence %.2f < %.2f, "
                "falling back to ML: %s (%.2f)",
                conf, self.threshold, fallback_verdict, ml_conf,
            )
            modified = dict(args)
            modified["verdict"] = fallback_verdict
            modified["attack_type"] = fallback_type
            modified["confidence"] = ml_conf
            modified["reasoning"] = (
                f"[LOW_CONF_FALLBACK] Agent confidence {conf:.2f} < {self.threshold}. "
                f"Using ML prediction: {ml_pred} ({ml_conf:.2f}). "
                f"Original reasoning: {args.get('reasoning', '')}"
            )
            return PermissionResult(
                allowed=True,
                modified_args=modified,
                violations=[f"low_confidence_fallback: {conf:.2f} < {self.threshold}"],
            )

        return PermissionResult(allowed=True)


class ConsistencyChecker:
    """Detects Agent/ML disagreement and gives Agent a retry chance."""

    def __init__(self, ml_confidence_threshold: float = 0.9):
        self.ml_confidence_threshold = ml_confidence_threshold

    def check(
        self,
        args: dict[str, Any],
        ml_prediction: Optional[dict] = None,
    ) -> PermissionResult:
        if not ml_prediction:
            return PermissionResult(allowed=True)

        agent_verdict = str(args.get("verdict", "")).lower()
        ml_pred = ml_prediction.get("prediction", "Normal")
        ml_is_attack = ml_pred.lower() not in ("normal", "benign")
        ml_conf = float(ml_prediction.get("confidence", 0.0))

        # Agent says benign but ML says attack with high confidence
        if agent_verdict == "benign" and ml_is_attack and ml_conf > self.ml_confidence_threshold:
            msg = (
                f"The ML anomaly detector strongly disagrees with your verdict. "
                f"It predicts '{ml_pred}' with {ml_conf:.0%} confidence. "
                f"Please reconsider your analysis and provide a final classify call."
            )
            logger.info(
                "ConsistencyChecker: Agent=benign but ML=%s (%.2f), requesting retry",
                ml_pred, ml_conf,
            )
            return PermissionResult(
                allowed=False,
                needs_retry=True,
                retry_message=msg,
                violations=[f"consistency_conflict: agent=benign, ml={ml_pred}@{ml_conf:.2f}"],
            )

        return PermissionResult(allowed=True)


class OutputLimiter:
    """Truncates long reasoning strings."""

    def __init__(self, max_chars: int = 500):
        self.max_chars = max_chars

    def check(self, args: dict[str, Any]) -> PermissionResult:
        reasoning = str(args.get("reasoning", ""))
        if len(reasoning) > self.max_chars:
            modified = dict(args)
            modified["reasoning"] = reasoning[: self.max_chars] + "..."
            return PermissionResult(
                allowed=True,
                modified_args=modified,
                violations=[f"reasoning_truncated: {len(reasoning)} > {self.max_chars}"],
            )
        return PermissionResult(allowed=True)


class PermissionsManager:
    """Combines all permission components.

    Usage:
        pm = PermissionsManager(enabled=True)
        result = pm.check_classify(classify_args, ml_prediction, already_retried)
    """

    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = 0.3,
        ml_conflict_threshold: float = 0.9,
        max_reasoning_chars: int = 500,
    ):
        self.enabled = enabled
        self.format_validator = FormatValidator()
        self.confidence_gate = ConfidenceGate(threshold=confidence_threshold)
        self.consistency_checker = ConsistencyChecker(
            ml_confidence_threshold=ml_conflict_threshold
        )
        self.output_limiter = OutputLimiter(max_chars=max_reasoning_chars)

    def check_classify(
        self,
        args: dict[str, Any],
        ml_prediction: Optional[dict] = None,
        already_retried: bool = False,
    ) -> PermissionResult:
        """Run all permission checks on a classify call.

        Args:
            args: classify tool arguments
            ml_prediction: Last check_anomaly result (if any)
            already_retried: If True, skip ConsistencyChecker (only one retry)

        Returns:
            PermissionResult with combined violations and modifications.
        """
        if not self.enabled:
            return PermissionResult(allowed=True)

        all_violations: list[str] = []
        current_args = dict(args)

        # 1. Format validation
        fmt_result = self.format_validator.check(current_args)
        if fmt_result.modified_args:
            current_args = fmt_result.modified_args
        all_violations.extend(fmt_result.violations)

        # 2. Consistency check (only if not already retried)
        if not already_retried:
            cons_result = self.consistency_checker.check(current_args, ml_prediction)
            if cons_result.needs_retry:
                cons_result.violations = all_violations + cons_result.violations
                return cons_result

        # 3. Confidence gate (after consistency, so retry has priority)
        conf_result = self.confidence_gate.check(current_args, ml_prediction)
        if conf_result.modified_args:
            current_args = conf_result.modified_args
        all_violations.extend(conf_result.violations)

        # 4. Output limiter
        out_result = self.output_limiter.check(current_args)
        if out_result.modified_args:
            current_args = out_result.modified_args
        all_violations.extend(out_result.violations)

        return PermissionResult(
            allowed=True,
            modified_args=current_args if all_violations else None,
            violations=all_violations,
        )
