"""
beta_agent_llm.py — Agent-Beta LLM variant (pattern-matching perspective).

DEAD CODE (audit 2026-07-28): no experiment ever instantiates this class
(v1 E4/E5 use BetaAgentML; grep shows only the package __init__ re-export).
Kept for reference; its prompt previously enumerated the full label set
including held-out classes, now trimmed to known classes only.

Uses a small model (e.g. llama3.2:1b) via Ollama with a pattern-matching
oriented prompt. Complements Agent-Alpha's semantic reasoning with fast,
statistical-feature-focused detection.
"""

from __future__ import annotations

import json
import re
import time
from typing import List, Optional

import requests

from .base_agent import AgentVerdict, BaseAgent

# ---------------------------------------------------------------------------
# Default prompt template — pattern-matching perspective
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a network intrusion detection system focused on PATTERN MATCHING.
Your job is to classify network traffic as benign or attack based on
statistical feature patterns. Do NOT reason about attack semantics or intent.
Instead, focus on:
- Abnormal numeric value combinations (e.g. high byte counts with zero packets)
- Known attack signatures in protocol/state/service fields
- Statistical outliers in traffic features

Respond ONLY with a JSON object:
{"verdict": "benign" or "attack", "attack_type": null or string, "confidence": 0.0-1.0, "reasoning": "brief pattern-based explanation"}
"""

USER_PROMPT_TEMPLATE = """\
Classify this network traffic record based on feature patterns:

{traffic_text}

Known attack types: {attack_types}
JSON response:"""

DEFAULT_ATTACK_TYPES = "Normal, Backdoor, DoS, Exploits, Reconnaissance"  # known classes only (held-out names removed, audit 2026-07-28)


class BetaAgentLLM(BaseAgent):
    """Agent-Beta: small LLM for pattern-matching-based detection via Ollama."""

    def __init__(
        self,
        name: str = "Beta-LLM",
        model: str = "llama3.2:1b",
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        attack_types: str = DEFAULT_ATTACK_TYPES,
        timeout: float = 30.0,
    ):
        super().__init__(name)
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.temperature = temperature
        self.attack_types = attack_types
        self.timeout = timeout

    def analyze(self, traffic_text: str) -> AgentVerdict:
        user_msg = USER_PROMPT_TEMPLATE.format(
            traffic_text=traffic_text,
            attack_types=self.attack_types,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 200,
            },
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            return AgentVerdict(
                verdict="benign",
                confidence=0.0,
                reasoning=f"Ollama request failed: {e}",
                agent_name=self.name,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        latency = (time.perf_counter() - t0) * 1000
        raw = data.get("message", {}).get("content", "")
        token_count = data.get("eval_count", 0)

        verdict = self._parse_response(raw)
        verdict.agent_name = self.name
        verdict.latency_ms = latency
        verdict.token_count = token_count
        return verdict

    def analyze_batch(self, traffic_texts: List[str]) -> List[AgentVerdict]:
        """Sequential batch — Ollama 1b model is fast enough per-sample."""
        return [self.analyze(t) for t in traffic_texts]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(raw: str) -> AgentVerdict:
        """Extract JSON verdict from LLM output, with fallback parsing."""
        # Try to find JSON in the response
        json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if json_match:
            try:
                d = json.loads(json_match.group())
                return AgentVerdict(
                    verdict="attack" if d.get("verdict", "").lower().startswith("attack") else "benign",
                    attack_type=d.get("attack_type"),
                    confidence=float(d.get("confidence", 0.5)),
                    reasoning=d.get("reasoning", ""),
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: keyword scan
        raw_lower = raw.lower()
        if "attack" in raw_lower:
            return AgentVerdict(verdict="attack", confidence=0.3, reasoning=f"Fallback parse: {raw[:200]}")
        return AgentVerdict(verdict="benign", confidence=0.3, reasoning=f"Fallback parse: {raw[:200]}")
