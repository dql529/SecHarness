"""
consensus.py — Consensus Module for SecHarness Dual-Agent Verification Loop.

Receives AgentVerdict from Agent-Alpha and Agent-Beta, applies configurable
consensus logic, and outputs a final detection decision with full audit trail.
This is the core contribution of the SecHarness framework (Section 3.2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

from .base_agent import AgentVerdict


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------
@dataclass
class ConsensusResult:
    """Final detection decision from the Dual-Agent Verification Loop."""

    final_verdict: Literal["benign", "attack", "escalate"]
    final_attack_type: Optional[str] = None
    consensus_type: Literal[
        "agreement", "alpha_only", "beta_only", "type_conflict", "uncertain"
    ] = "agreement"
    combined_confidence: float = 0.0
    alpha_verdict: Optional[AgentVerdict] = None
    beta_verdict: Optional[AgentVerdict] = None
    escalation_reason: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def is_disagreement(self) -> bool:
        return self.consensus_type in ("alpha_only", "beta_only", "type_conflict", "uncertain")


# ---------------------------------------------------------------------------
# Consensus Module
# ---------------------------------------------------------------------------
class ConsensusModule:
    """Dual-Agent Verification Loop consensus logic.

    Parameters
    ----------
    alpha_weight : float
        Weight for Agent-Alpha in confidence fusion (default 0.6).
    beta_weight : float
        Weight for Agent-Beta in confidence fusion (default 0.4).
    disagreement_strategy : str
        How to resolve disagreements:
        - "alpha_priority": trust Alpha when agents disagree
        - "beta_priority": trust Beta when agents disagree
        - "escalate_all": escalate all disagreements for review
    confidence_threshold : float
        Below this value, a verdict is considered low-confidence (default 0.4).
    """

    def __init__(
        self,
        alpha_weight: float = 0.6,
        beta_weight: float = 0.4,
        disagreement_strategy: Literal[
            "alpha_priority", "beta_priority", "escalate_all"
        ] = "escalate_all",
        confidence_threshold: float = 0.4,
    ):
        assert abs(alpha_weight + beta_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight
        self.disagreement_strategy = disagreement_strategy
        self.confidence_threshold = confidence_threshold

    def resolve(
        self, alpha: AgentVerdict, beta: AgentVerdict
    ) -> ConsensusResult:
        """Apply consensus logic to a pair of agent verdicts."""
        t0 = time.perf_counter()

        combined_conf = (
            self.alpha_weight * alpha.confidence
            + self.beta_weight * beta.confidence
        )

        # --- Both low confidence → uncertain ---
        if (
            alpha.confidence < self.confidence_threshold
            and beta.confidence < self.confidence_threshold
        ):
            result = ConsensusResult(
                final_verdict="escalate",
                consensus_type="uncertain",
                combined_confidence=combined_conf,
                alpha_verdict=alpha,
                beta_verdict=beta,
                escalation_reason=(
                    f"Both agents below confidence threshold "
                    f"({alpha.confidence:.2f}, {beta.confidence:.2f} < {self.confidence_threshold})"
                ),
            )
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        # --- Agreement path ---
        if alpha.verdict == beta.verdict:
            if alpha.verdict == "benign":
                result = ConsensusResult(
                    final_verdict="benign",
                    consensus_type="agreement",
                    combined_confidence=combined_conf,
                    alpha_verdict=alpha,
                    beta_verdict=beta,
                )
            else:
                # Both attack — check attack_type consistency
                alpha_type = (alpha.attack_type or "").lower().strip()
                beta_type = (beta.attack_type or "").lower().strip()

                if alpha_type == beta_type or not alpha_type or not beta_type:
                    # Same type or one didn't specify → agree
                    result = ConsensusResult(
                        final_verdict="attack",
                        final_attack_type=alpha.attack_type or beta.attack_type,
                        consensus_type="agreement",
                        combined_confidence=combined_conf,
                        alpha_verdict=alpha,
                        beta_verdict=beta,
                    )
                else:
                    # Both say attack but different types → type_conflict
                    result = ConsensusResult(
                        final_verdict="attack",
                        final_attack_type=alpha.attack_type,  # Alpha priority for type
                        consensus_type="type_conflict",
                        combined_confidence=combined_conf,
                        alpha_verdict=alpha,
                        beta_verdict=beta,
                        escalation_reason=(
                            f"Attack type conflict: Alpha={alpha.attack_type}, Beta={beta.attack_type}"
                        ),
                    )

            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        # --- Disagreement path ---
        if alpha.verdict == "attack":
            tag = "alpha_only"
            reason = "Alpha=attack, Beta=benign — possible LLM hallucination or novel attack"
        else:
            tag = "beta_only"
            reason = "Alpha=benign, Beta=attack — possible known pattern variant"

        # Apply disagreement strategy
        if self.disagreement_strategy == "alpha_priority":
            final = alpha.verdict
            final_type = alpha.attack_type if alpha.verdict == "attack" else None
        elif self.disagreement_strategy == "beta_priority":
            final = beta.verdict
            final_type = beta.attack_type if beta.verdict == "attack" else None
        else:  # escalate_all
            final = "escalate"
            final_type = (
                alpha.attack_type if alpha.verdict == "attack" else beta.attack_type
            )

        result = ConsensusResult(
            final_verdict=final,
            final_attack_type=final_type,
            consensus_type=tag,
            combined_confidence=combined_conf,
            alpha_verdict=alpha,
            beta_verdict=beta,
            escalation_reason=reason,
        )
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def resolve_batch(
        self,
        alphas: list[AgentVerdict],
        betas: list[AgentVerdict],
    ) -> list[ConsensusResult]:
        """Resolve a batch of verdict pairs."""
        assert len(alphas) == len(betas), "Alpha and Beta batch sizes must match"
        return [self.resolve(a, b) for a, b in zip(alphas, betas)]

    def __repr__(self) -> str:
        return (
            f"<ConsensusModule weights=({self.alpha_weight}/{self.beta_weight}) "
            f"strategy={self.disagreement_strategy!r} threshold={self.confidence_threshold}>"
        )
