"""
alpha_agent_vllm.py — Agent-Alpha using vLLM HTTP backend (OpenAI-compatible API).

Calls the vLLM /v1/chat/completions endpoint with dynamic LoRA adapter support.
Imports SYSTEM_PROMPT and _parse_json_response from alpha_agent_hf to prevent
regex/prompt drift between backends (single source of truth).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from project.src.agents.alpha_agent_hf import SYSTEM_PROMPT, _parse_json_response
from project.src.agents.base_agent import BaseAgent, AgentVerdict

logger = logging.getLogger(__name__)

# Exponential backoff delays for retries (seconds)
_BACKOFF_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)


class AlphaAgentVLLM(BaseAgent):
    """Agent-Alpha using vLLM OpenAI-compatible HTTP endpoint.

    CONTRACT: AlphaAgentVLLM.__init__
      inputs:
        endpoint: str — base URL, e.g. "http://127.0.0.1:8000/v1"
        model_name: str — LoRA adapter alias served by vLLM
        temperature: float — sampling temperature in [0.0, 1.0]
        max_new_tokens: int — maximum tokens to generate (> 0)
        timeout_s: float — per-request timeout in seconds (> 0)
        max_retries: int — retry attempts on transient errors [0, 10]
      output: None (raises ValueError on invalid params)
      preconditions: none (endpoint not checked at init)
      error modes: ValueError if temperature not in [0.0, 1.0]
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/v1",
        model_name: str = "alpha_unsw_v2_clean",
        temperature: float = 0.1,
        max_new_tokens: int = 150,
        timeout_s: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(name="Agent-Alpha-vLLM")
        if not (0.0 <= temperature <= 1.0):
            raise ValueError(f"temperature must be in [0.0, 1.0], got {temperature}")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if not (0 <= max_retries <= 10):
            raise ValueError(f"max_retries must be in [0, 10], got {max_retries}")

        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._chat_url = f"{self.endpoint}/chat/completions"

        logger.info(
            "AlphaAgentVLLM ready: endpoint=%s model=%s temperature=%.2f",
            self.endpoint,
            self.model_name,
            self.temperature,
        )

    def _build_payload(self, traffic_text: str) -> dict:
        """Build the JSON payload for /v1/chat/completions.

        CONTRACT: _build_payload
          inputs: traffic_text:str — raw kv-format traffic record
          output: dict — OpenAI-compatible chat completion request body
          preconditions: traffic_text is non-empty
          error modes: none
        """
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": traffic_text},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }

    def analyze(self, traffic_text: str) -> AgentVerdict:
        """Analyze a single traffic record via vLLM HTTP endpoint.

        CONTRACT: analyze
          inputs: traffic_text:str — kv-format network traffic record
          output: AgentVerdict — structured verdict with confidence, reasoning
          preconditions: vLLM server reachable at self.endpoint
          error modes:
            - requests.Timeout → retry with exponential backoff
            - HTTP 5xx → retry with exponential backoff
            - parse fail after all retries → sentinel verdict:
                verdict="benign", confidence=0.0,
                reasoning="PARSE_FAIL: <raw content prefix>"
            Do NOT raise — must not crash the batch.
        """
        t0 = time.perf_counter()
        payload = self._build_payload(traffic_text)
        last_content: Optional[str] = None
        resp_json: dict = {}

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                delay = _BACKOFF_DELAYS[min(attempt - 1, len(_BACKOFF_DELAYS) - 1)]
                logger.warning(
                    "Retry %d/%d after %.0fs (model=%s)",
                    attempt,
                    self.max_retries,
                    delay,
                    self.model_name,
                )
                time.sleep(delay)

            try:
                resp = requests.post(
                    self._chat_url,
                    json=payload,
                    timeout=self.timeout_s,
                )
                if resp.status_code >= 500:
                    logger.warning(
                        "HTTP %d from vLLM endpoint (attempt %d)",
                        resp.status_code,
                        attempt,
                    )
                    continue
                resp.raise_for_status()
                resp_json = resp.json()
                last_content = resp_json["choices"][0]["message"]["content"]

                parsed = _parse_json_response(last_content)
                if parsed and "verdict" in parsed:
                    verdict_str = str(parsed["verdict"]).lower().strip()
                    if verdict_str not in ("benign", "attack"):
                        verdict_str = "benign"
                    latency_ms = (time.perf_counter() - t0) * 1000
                    return AgentVerdict(
                        verdict=verdict_str,  # type: ignore[arg-type]
                        attack_type=parsed.get("attack_type"),
                        confidence=float(parsed.get("confidence", 0.5)),
                        reasoning=str(parsed.get("reasoning", ""))[:500],
                        agent_name="alpha_vllm",
                        latency_ms=latency_ms,
                        token_count=resp_json.get("usage", {}).get("completion_tokens", 0),
                    )
                # Parse returned None — retry
                logger.warning(
                    "Parse failed on attempt %d (content prefix: %s)",
                    attempt,
                    (last_content or "")[:80],
                )

            except requests.Timeout:
                logger.warning(
                    "Request timed out on attempt %d (timeout=%.0fs)",
                    attempt,
                    self.timeout_s,
                )
            except requests.RequestException as exc:
                logger.warning("Request error on attempt %d: %s", attempt, exc)

        # All retries exhausted — return sentinel parse-fail verdict
        latency_ms = (time.perf_counter() - t0) * 1000
        fail_content = (last_content or "")[:200]
        logger.error(
            "All %d retries exhausted; returning PARSE_FAIL sentinel (model=%s)",
            self.max_retries,
            self.model_name,
        )
        return AgentVerdict(
            verdict="benign",
            confidence=0.0,
            reasoning=f"PARSE_FAIL: {fail_content}",
            agent_name="alpha_vllm",
            latency_ms=latency_ms,
        )

    def __repr__(self) -> str:
        return (
            f"<AlphaAgentVLLM endpoint={self.endpoint!r} "
            f"model={self.model_name!r} temperature={self.temperature}>"
        )
