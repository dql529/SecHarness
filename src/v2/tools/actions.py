"""
actions.py — Action interface tools: classify, escalate, log_decision.

These are terminal or side-effect tools that don't return analysis data.
"""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

CLASSIFY_SCHEMA = {
    "name": "classify",
    "description": (
        "Submit final classification verdict. This terminates the analysis. "
        "Use after gathering enough evidence from other tools."
    ),
    "parameters": {
        "verdict": {
            "type": "string",
            "enum": ["benign", "attack"],
            "description": "Final classification: benign or attack",
        },
        "attack_type": {
            "type": "string",
            "nullable": True,
            "description": "Attack category (e.g., DoS, Exploits, Recon). Required if verdict=attack.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence score between 0.0 and 1.0",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief reasoning chain for this verdict",
        },
    },
}

ESCALATE_SCHEMA = {
    "name": "escalate",
    "description": (
        "Escalate to human analyst when uncertain. Terminates analysis."
    ),
    "parameters": {
        "reason": {"type": "string", "description": "Why escalation is needed"},
        "partial_verdict": {
            "type": "string",
            "enum": ["benign", "attack", "unknown"],
        },
        "confidence": {"type": "number"},
    },
}

LOG_DECISION_SCHEMA = {
    "name": "log_decision",
    "description": "Log an intermediate reasoning step or observation.",
    "parameters": {
        "observation": {"type": "string", "description": "What the agent noticed"},
        "implication": {"type": "string", "description": "What this means for the verdict"},
    },
}


def handle_escalate(reason: str = "", partial_verdict: str = "unknown", confidence: float = 0.3, **kwargs) -> str:
    """Handle escalate call — record but auto-fallback to classify."""
    return json.dumps({
        "escalated": True,
        "reason": reason,
        "partial_verdict": partial_verdict,
        "confidence": confidence,
    })


def handle_log_decision(observation: str = "", implication: str = "", **kwargs) -> str:
    """Handle log_decision — just acknowledge and record."""
    return json.dumps({
        "logged": True,
        "timestamp": time.time(),
    })
