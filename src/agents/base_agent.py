"""
base_agent.py — Abstract base class for SecHarness detection agents.

Both Agent-Alpha (behavioral analysis) and Agent-Beta (pattern matching)
implement this interface so the Consensus Module can treat them uniformly.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional


@dataclass
class AgentVerdict:
    """Structured output from a detection agent."""

    verdict: Literal["benign", "attack"]
    attack_type: Optional[str] = None
    confidence: float = 0.5
    reasoning: str = ""
    # Metadata — filled by the agent after inference
    agent_name: str = ""
    latency_ms: float = 0.0
    token_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "AgentVerdict":
        return AgentVerdict(**{k: v for k, v in d.items() if k in AgentVerdict.__dataclass_fields__})


class BaseAgent(ABC):
    """Abstract base for all SecHarness detection agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, traffic_text: str) -> AgentVerdict:
        """Analyze a single traffic record (kv text) and return a verdict."""
        ...

    def analyze_batch(self, traffic_texts: List[str]) -> List[AgentVerdict]:
        """Analyze a batch of traffic records.

        Default: sequential calls to `analyze`. Subclasses may override
        for true batching (e.g. concurrent Ollama requests).
        """
        results = []
        for text in traffic_texts:
            t0 = time.perf_counter()
            v = self.analyze(text)
            v.latency_ms = (time.perf_counter() - t0) * 1000
            v.agent_name = self.name
            results.append(v)
        return results

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
