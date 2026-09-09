"""
audit.py — AuditRecordV2 + AuditLoggerV2 for SecHarness v2.

Records the complete tool-call chain and LLM reasoning for each sample.
Supports JSONL streaming, in-memory caching, and aggregation queries.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """A single tool invocation within the agent loop."""
    step: int
    tool: str
    input: dict[str, Any]
    output: str  # JSON string returned to Agent
    latency_ms: float = 0.0
    timestamp: float = 0.0


@dataclass
class LLMCallRecord:
    """A single LLM generation within the agent loop."""
    step: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw_output: str = ""


@dataclass
class AuditRecordV2:
    """Complete audit trail for one traffic sample in v2."""
    # Input
    sample_index: int = 0
    traffic_text: str = ""
    ground_truth_label: str = ""
    is_zeroday: bool = False

    # Agent config
    model: str = ""
    adapter: str = ""
    harness_enabled: bool = False
    temperature: float = 0.1
    max_steps: int = 5

    # Traces
    tool_chain: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)

    # Result
    verdict: str = ""  # "benign" | "attack"
    attack_type: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    termination: str = ""  # "classify" | "max_steps" | "parse_fail" | "direct_verdict"
    total_steps: int = 0
    tools_used: list[str] = field(default_factory=list)
    knowledge_loaded: list[str] = field(default_factory=list)
    self_corrected: bool = False
    escalated: bool = False
    low_confidence: bool = False

    # Evaluation (filled post-hoc)
    detection_correct: Optional[bool] = None
    attack_type_correct: Optional[bool] = None
    binary_pred: str = ""
    binary_gt: str = ""

    # Efficiency
    total_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    num_llm_calls: int = 0

    # Metadata
    timestamp: float = field(default_factory=time.time)
    version: str = "2.0"

    def compute_efficiency(self) -> None:
        """Populate efficiency fields from tool_chain and llm_calls."""
        self.llm_latency_ms = sum(c.latency_ms for c in self.llm_calls)
        self.tool_latency_ms = sum(c.latency_ms for c in self.tool_chain)
        self.total_latency_ms = self.llm_latency_ms + self.tool_latency_ms
        self.total_input_tokens = sum(c.input_tokens for c in self.llm_calls)
        self.total_output_tokens = sum(c.output_tokens for c in self.llm_calls)
        self.total_tokens = self.total_input_tokens + self.total_output_tokens
        self.num_llm_calls = len(self.llm_calls)
        self.total_steps = len(self.tool_chain)
        self.tools_used = list(dict.fromkeys(c.tool for c in self.tool_chain))

    def evaluate(self, ground_truth: str) -> None:
        """Fill evaluation fields given ground truth label."""
        self.ground_truth_label = ground_truth
        gt_is_attack = ground_truth.lower() not in ("normal", "benign")
        pred_is_attack = self.verdict == "attack"

        self.binary_gt = "attack" if gt_is_attack else "benign"
        self.binary_pred = self.verdict
        self.detection_correct = (pred_is_attack == gt_is_attack)

        if pred_is_attack and gt_is_attack and self.attack_type:
            self.attack_type_correct = (
                self.attack_type.lower() == ground_truth.lower()
            )
        else:
            self.attack_type_correct = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "sample_index": self.sample_index,
            "input": {
                "traffic_text": self.traffic_text[:500],
                "ground_truth_label": self.ground_truth_label,
                "is_zeroday": self.is_zeroday,
            },
            "agent_config": {
                "model": self.model,
                "adapter": self.adapter,
                "harness_enabled": self.harness_enabled,
                "temperature": self.temperature,
                "max_steps": self.max_steps,
            },
            "tool_chain": [
                {
                    "step": tc.step, "tool": tc.tool,
                    "input": tc.input, "output": tc.output,
                    "latency_ms": tc.latency_ms, "timestamp": tc.timestamp,
                }
                for tc in self.tool_chain
            ],
            "llm_calls": [
                {
                    "step": lc.step,
                    "input_tokens": lc.input_tokens,
                    "output_tokens": lc.output_tokens,
                    "latency_ms": lc.latency_ms,
                    "raw_output": lc.raw_output[:300],
                }
                for lc in self.llm_calls
            ],
            "result": {
                "verdict": self.verdict,
                "attack_type": self.attack_type,
                "confidence": self.confidence,
                "reasoning": self.reasoning[:500],
                "termination": self.termination,
                "total_steps": self.total_steps,
                "tools_used": self.tools_used,
                "knowledge_loaded": self.knowledge_loaded,
                "self_corrected": self.self_corrected,
                "escalated": self.escalated,
                "low_confidence": self.low_confidence,
            },
            "evaluation": {
                "detection_correct": self.detection_correct,
                "attack_type_correct": self.attack_type_correct,
                "binary_pred": self.binary_pred,
                "binary_gt": self.binary_gt,
            },
            "efficiency": {
                "total_latency_ms": round(self.total_latency_ms, 2),
                "llm_latency_ms": round(self.llm_latency_ms, 2),
                "tool_latency_ms": round(self.tool_latency_ms, 2),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "num_llm_calls": self.num_llm_calls,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AuditLoggerV2:
    """JSONL streaming writer + in-memory cache for v2 audit records."""

    def __init__(self, log_path: Optional[str | Path] = None):
        self._records: list[AuditRecordV2] = []
        self._log_path = Path(log_path) if log_path else None
        self._file = None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._log_path, "a", encoding="utf-8")

    def log(self, record: AuditRecordV2) -> None:
        """Write one record to JSONL + memory cache."""
        self._records.append(record)
        if self._file:
            self._file.write(record.to_json() + "\n")
            self._file.flush()

    def log_batch(self, records: list[AuditRecordV2]) -> None:
        for r in records:
            self.log(r)

    @property
    def records(self) -> list[AuditRecordV2]:
        return self._records

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ---- Aggregation queries ----

    def tool_usage_stats(self) -> dict[str, int]:
        """Count how often each tool was called across all records."""
        counts: dict[str, int] = {}
        for r in self._records:
            for tc in r.tool_chain:
                counts[tc.tool] = counts.get(tc.tool, 0) + 1
        return counts

    def reasoning_chain_stats(self) -> dict[str, float]:
        """Average number of LLM calls and tool calls per sample."""
        if not self._records:
            return {"avg_llm_calls": 0, "avg_tool_calls": 0, "avg_steps": 0}
        n = len(self._records)
        return {
            "avg_llm_calls": sum(r.num_llm_calls for r in self._records) / n,
            "avg_tool_calls": sum(len(r.tool_chain) for r in self._records) / n,
            "avg_steps": sum(r.total_steps for r in self._records) / n,
        }

    def self_correction_rate(self) -> float:
        """Fraction of samples where Agent verdict differs from check_anomaly prediction.

        Self-correction is defined as the Agent disagreeing with the ML tool.
        """
        relevant = 0
        disagreed = 0
        for r in self._records:
            # Find check_anomaly calls
            for tc in r.tool_chain:
                if tc.tool == "check_anomaly":
                    try:
                        ml_result = json.loads(tc.output)
                        ml_pred = ml_result.get("prediction", "").lower()
                        ml_is_attack = ml_pred not in ("normal", "benign", "")
                        agent_is_attack = r.verdict == "attack"
                        relevant += 1
                        if ml_is_attack != agent_is_attack:
                            disagreed += 1
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    break  # Only first check_anomaly matters
        return disagreed / relevant if relevant > 0 else 0.0

    def escalation_rate(self) -> float:
        """Fraction of samples that were escalated."""
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.escalated) / len(self._records)

    def fast_path_rate(self) -> float:
        """Fraction of samples completed in <= 3 tool calls (fast path)."""
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.total_steps <= 3) / len(self._records)

    def efficiency_stats(self) -> dict[str, float]:
        """Average latency and token usage."""
        if not self._records:
            return {}
        n = len(self._records)
        return {
            "avg_total_latency_ms": sum(r.total_latency_ms for r in self._records) / n,
            "avg_llm_latency_ms": sum(r.llm_latency_ms for r in self._records) / n,
            "avg_total_tokens": sum(r.total_tokens for r in self._records) / n,
            "avg_input_tokens": sum(r.total_input_tokens for r in self._records) / n,
            "avg_output_tokens": sum(r.total_output_tokens for r in self._records) / n,
        }

    @staticmethod
    def load_from_file(path: str | Path) -> list[dict]:
        """Read all JSONL records from a file as dicts."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
