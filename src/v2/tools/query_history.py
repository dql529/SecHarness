"""
query_history.py — Sliding window history query tool.

Maintains recent detection records and matches by (proto, service, state).
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import dataclass
from typing import Optional

from ...agents.beta_agent_ml import parse_kv_text

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "query_history",
    "description": (
        "Query recent detection history for related traffic "
        "(same protocol/service/state pattern). "
        "Returns recent verdicts and attack ratio."
    ),
    "parameters": {
        "traffic_text": {
            "type": "string",
            "description": "Raw kv-format traffic record",
        }
    },
}


@dataclass
class HistoryEntry:
    """A single detection record in history."""
    key: tuple[str, str, str]  # (proto, service, state)
    verdict: str
    attack_type: Optional[str]
    confidence: float
    index: int  # position in processing sequence


class QueryHistoryTool:
    """Sliding window history with (proto, service, state) matching."""

    def __init__(self, maxlen: int = 200):
        self._history: collections.deque[HistoryEntry] = collections.deque(maxlen=maxlen)
        self._counter: int = 0

    def __call__(self, traffic_text: str) -> str:
        """Query history for traffic matching the same (proto, service, state).

        Returns:
            JSON string with related samples count, attack ratio, recent verdicts.
        """
        features = parse_kv_text(traffic_text)
        query_key = (
            features.get("proto", ""),
            features.get("service", ""),
            features.get("state", ""),
        )

        related = []
        for entry in self._history:
            if entry.key == query_key:
                related.append(entry)

        # Recent N matching entries
        recent = related[-5:] if related else []
        attack_count = sum(1 for e in related if e.verdict == "attack")
        attack_ratio = attack_count / len(related) if related else 0.0

        recent_verdicts = [
            {
                "index": e.index,
                "verdict": e.verdict,
                "type": e.attack_type,
                "confidence": round(e.confidence, 2),
            }
            for e in recent
        ]

        result = {
            "related_samples": len(related),
            "attack_ratio": round(attack_ratio, 3),
            "recent_verdicts": recent_verdicts,
            "pattern_note": (
                f"{attack_count}/{len(related)} recent similar traffic flagged as attacks"
                if related else "No matching history found"
            ),
        }
        return json.dumps(result, ensure_ascii=False)

    def push(
        self,
        traffic_text: str,
        verdict: str,
        attack_type: Optional[str] = None,
        confidence: float = 0.5,
    ) -> None:
        """Add a new detection record to history."""
        features = parse_kv_text(traffic_text)
        key = (
            features.get("proto", ""),
            features.get("service", ""),
            features.get("state", ""),
        )
        self._history.append(HistoryEntry(
            key=key,
            verdict=verdict,
            attack_type=attack_type,
            confidence=confidence,
            index=self._counter,
        ))
        self._counter += 1

    def reset(self) -> None:
        """Clear history (call at experiment/dataset start)."""
        self._history.clear()
        self._counter = 0
