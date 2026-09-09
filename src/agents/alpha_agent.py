"""
alpha_agent.py — Agent-Alpha: behavioral analysis via local LLM (Ollama).

Perspective: session-level behavioral semantics.
  - "Is the behavioral pattern in this session anomalous?"
  - "Does this combination of traffic features match known attack behaviors?"

Uses Ollama HTTP API (localhost:11434) for local LLM inference.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests

from .base_agent import BaseAgent, AgentVerdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — guides the LLM to reason from a behavioral perspective
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a network intrusion detection expert specializing in behavioral analysis.

Your task: analyze a network traffic record (key=value format) and determine whether it represents benign activity or an attack.

Analysis approach — think step by step:
1. Parse each feature and its value (low/medium/high/very_high/zero/missing).
2. Identify behavioral anomalies: unusual protocol-state-service combinations, \
extreme byte/packet ratios, suspicious timing patterns, abnormal connection counts.
3. Consider what attack behaviors these features could indicate \
(e.g., high sbytes + zero dbytes → data exfiltration; many connections + low duration → scanning).
4. Make your verdict with a calibrated confidence score.

Respond ONLY with a JSON object (no markdown, no extra text):
{
  "verdict": "benign" or "attack",
  "attack_type": null or a short attack category name (e.g., "DoS", "Reconnaissance", "Exploits"),
  "confidence": a float between 0.0 and 1.0,
  "reasoning": "brief explanation of your analysis (1-3 sentences)"
}"""

USER_TEMPLATE = "Analyze this network traffic record:\n{traffic_text}"


def _parse_truncated_json(content: str) -> dict | None:
    """Try to extract fields from truncated JSON when the response was cut off."""
    # Look for the opening brace
    start = content.find("{")
    if start < 0:
        return None
    fragment = content[start:]
    result = {}
    # Extract "verdict"
    m = re.search(r'"verdict"\s*:\s*"(benign|attack)"', fragment, re.IGNORECASE)
    if m:
        result["verdict"] = m.group(1).lower()
    # Extract "attack_type"
    m = re.search(r'"attack_type"\s*:\s*"([^"]*)"', fragment)
    if m:
        result["attack_type"] = m.group(1)
    elif re.search(r'"attack_type"\s*:\s*null', fragment):
        result["attack_type"] = None
    # Extract "confidence"
    m = re.search(r'"confidence"\s*:\s*([0-9.]+)', fragment)
    if m:
        result["confidence"] = m.group(1)
    # Extract "reasoning" (may be truncated)
    m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', fragment)
    if m:
        result["reasoning"] = m.group(1)
    return result if "verdict" in result else None


def _parse_kv_response(content: str) -> dict | None:
    """Parse key=value style responses from fine-tuned models (non-JSON fallback)."""
    result = {}
    # Match verdict=benign or verdict=attack anywhere in text
    m = re.search(r'verdict\s*[=:]\s*(benign|attack)', content, re.IGNORECASE)
    if m:
        result["verdict"] = m.group(1).lower()
    # Match attack_type
    m = re.search(r'attack_type\s*[=:]\s*(\w+)', content, re.IGNORECASE)
    if m and m.group(1).lower() != "null" and m.group(1).lower() != "none":
        result["attack_type"] = m.group(1)
    else:
        result["attack_type"] = None
    # Match confidence
    m = re.search(r'confidence\s*[=:]\s*([0-9.]+)', content, re.IGNORECASE)
    if m:
        result["confidence"] = m.group(1)
    # Extract reasoning — use remaining text
    m = re.search(r'reasoning\s*[=:]\s*(.+)', content, re.IGNORECASE)
    if m:
        result["reasoning"] = m.group(1).strip()
    else:
        # Use any sentence that looks like analysis
        lines = [l.strip() for l in content.split('\n') if len(l.strip()) > 20 and 'verdict' not in l.lower()]
        if lines:
            result["reasoning"] = lines[0][:200]
    # Detect attack keywords even if verdict not explicitly stated
    if "verdict" not in result:
        attack_kw = ["attack", "malicious", "exploit", "dos", "recon", "backdoor", "intrusion", "suspicious"]
        benign_kw = ["benign", "normal", "legitimate", "no attack", "not an attack"]
        content_lower = content.lower()
        if any(kw in content_lower for kw in attack_kw) and not any(kw in content_lower for kw in benign_kw):
            result["verdict"] = "attack"
        elif any(kw in content_lower for kw in benign_kw):
            result["verdict"] = "benign"
    return result if "verdict" in result else None


# ---------------------------------------------------------------------------
# AlphaAgent
# ---------------------------------------------------------------------------
class AlphaAgent(BaseAgent):
    """Behavioral analysis agent powered by a local LLM via Ollama."""

    def __init__(
        self,
        model: str = "llama3:8b",
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 256,
        timeout: float = 60.0,
        max_workers: int = 4,
    ):
        super().__init__(name="Agent-Alpha")
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def _call_ollama(self, traffic_text: str) -> dict:
        """Send a single inference request to Ollama and return raw response."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(traffic_text=traffic_text)},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_response(raw: dict) -> AgentVerdict:
        """Parse Ollama chat response into an AgentVerdict."""
        content = raw.get("message", {}).get("content", "")
        token_count = raw.get("eval_count", 0) + raw.get("prompt_eval_count", 0)

        # Try to extract JSON from the response (handle markdown fences)
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                verdict = parsed.get("verdict", "benign").lower().strip()
                if verdict not in ("benign", "attack"):
                    verdict = "benign"
                return AgentVerdict(
                    verdict=verdict,
                    attack_type=parsed.get("attack_type"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    reasoning=str(parsed.get("reasoning", "")),
                    token_count=token_count,
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: try to parse truncated JSON (when max_tokens cuts off the response)
        truncated = _parse_truncated_json(content)
        if truncated:
            verdict = truncated.get("verdict", "benign").lower().strip()
            if verdict not in ("benign", "attack"):
                verdict = "benign"
            return AgentVerdict(
                verdict=verdict,
                attack_type=truncated.get("attack_type"),
                confidence=float(truncated.get("confidence", 0.5)),
                reasoning=str(truncated.get("reasoning", "[truncated]")),
                token_count=token_count,
            )

        # Fallback: try kv-style parsing (fine-tuned model may output verdict=attack format)
        kv_parsed = _parse_kv_response(content)
        if kv_parsed:
            verdict = kv_parsed.get("verdict", "benign").lower().strip()
            if verdict not in ("benign", "attack"):
                verdict = "benign"
            return AgentVerdict(
                verdict=verdict,
                attack_type=kv_parsed.get("attack_type"),
                confidence=float(kv_parsed.get("confidence", 0.5)),
                reasoning=str(kv_parsed.get("reasoning", "[kv-parsed]")),
                token_count=token_count,
            )

        # Fallback: could not parse structured output
        logger.warning("Failed to parse LLM response, falling back to benign: %s", content[:200])
        return AgentVerdict(
            verdict="benign",
            confidence=0.0,
            reasoning=f"[PARSE_FAIL] raw: {content[:300]}",
            token_count=token_count,
        )

    def analyze(self, traffic_text: str) -> AgentVerdict:
        """Analyze a single traffic record."""
        t0 = time.perf_counter()
        try:
            raw = self._call_ollama(traffic_text)
            verdict = self._parse_response(raw)
        except requests.RequestException as e:
            logger.error("Ollama request failed: %s", e)
            verdict = AgentVerdict(
                verdict="benign", confidence=0.0,
                reasoning=f"[ERROR] Ollama request failed: {e}",
            )
        verdict.agent_name = self.name
        verdict.latency_ms = (time.perf_counter() - t0) * 1000
        return verdict

    def analyze_batch(self, traffic_texts: List[str]) -> List[AgentVerdict]:
        """Analyze a batch concurrently using a thread pool.

        Ollama processes one request at a time by default, but pipelining
        requests via threads hides network/scheduling overhead and keeps
        the GPU fed when Ollama's `OLLAMA_NUM_PARALLEL` > 1.
        """
        results: List[Optional[AgentVerdict]] = [None] * len(traffic_texts)

        def _work(idx: int, text: str) -> tuple[int, AgentVerdict]:
            v = self.analyze(text)
            return idx, v

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_work, i, t): i for i, t in enumerate(traffic_texts)}
            for future in as_completed(futures):
                idx, verdict = future.result()
                results[idx] = verdict

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def check_health(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            base_name = self.model.split(":")[0]
            available = any(base_name in m for m in models)
            if not available:
                logger.warning("Model %s not found. Available: %s", self.model, models)
            return available
        except requests.RequestException as e:
            logger.error("Ollama health check failed: %s", e)
            return False

    def __repr__(self) -> str:
        return f"<AlphaAgent model={self.model!r} url={self.ollama_url!r}>"
