"""
observability.py — Audit trail recording for SecHarness.

Records every detection decision as a structured AuditRecord, streams to JSONL,
and provides aggregation utilities for downstream analysis (RQ1-RQ4).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from ..agents.base_agent import AgentVerdict
from ..agents.consensus import ConsensusResult


# ---------------------------------------------------------------------------
# AuditRecord
# ---------------------------------------------------------------------------

@dataclass
class LatencyBreakdown:
    alpha_ms: float = 0.0
    beta_ms: float = 0.0
    consensus_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TokenUsage:
    alpha_tokens: int = 0
    beta_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.alpha_tokens + self.beta_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


@dataclass
class AuditRecord:
    """Complete audit trail for a single traffic sample."""

    timestamp: float  # epoch seconds
    sample_index: int
    traffic_text: str  # original kv text (may be truncated)

    alpha_verdict: AgentVerdict
    beta_verdict: AgentVerdict
    consensus_result: ConsensusResult

    ground_truth_label: Optional[str] = None  # e.g. "Normal", "DoS", ...
    detection_correct: Optional[bool] = None

    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    # --- serialization ---

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "sample_index": self.sample_index,
            "traffic_text": self.traffic_text,
            "alpha_verdict": self.alpha_verdict.to_dict(),
            "beta_verdict": self.beta_verdict.to_dict(),
            "consensus_result": self.consensus_result.to_dict(),
            "ground_truth_label": self.ground_truth_label,
            "detection_correct": self.detection_correct,
            "latency": self.latency.to_dict(),
            "token_usage": self.token_usage.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict) -> "AuditRecord":
        # Reconstruct ConsensusResult — it embeds alpha/beta verdicts,
        # but we store them separately; strip nested verdicts for reconstruction
        cr_data = d["consensus_result"]
        cr = ConsensusResult(
            final_verdict=cr_data["final_verdict"],
            final_attack_type=cr_data.get("final_attack_type"),
            consensus_type=cr_data.get("consensus_type", "agreement"),
            combined_confidence=cr_data.get("combined_confidence", 0.0),
            escalation_reason=cr_data.get("escalation_reason"),
            latency_ms=cr_data.get("latency_ms", 0.0),
        )
        return AuditRecord(
            timestamp=d["timestamp"],
            sample_index=d["sample_index"],
            traffic_text=d["traffic_text"],
            alpha_verdict=AgentVerdict.from_dict(d["alpha_verdict"]),
            beta_verdict=AgentVerdict.from_dict(d["beta_verdict"]),
            consensus_result=cr,
            ground_truth_label=d.get("ground_truth_label"),
            detection_correct=d.get("detection_correct"),
            latency=LatencyBreakdown(**d["latency"]) if "latency" in d else LatencyBreakdown(),
            token_usage=TokenUsage(
                alpha_tokens=d.get("token_usage", {}).get("alpha_tokens", 0),
                beta_tokens=d.get("token_usage", {}).get("beta_tokens", 0),
            ),
        )


# ---------------------------------------------------------------------------
# Helper: build an AuditRecord from agent outputs
# ---------------------------------------------------------------------------

def build_audit_record(
    sample_index: int,
    traffic_text: str,
    alpha_verdict: AgentVerdict,
    beta_verdict: AgentVerdict,
    consensus_result: ConsensusResult,
    ground_truth_label: Optional[str] = None,
    max_traffic_len: int = 500,
) -> AuditRecord:
    """Convenience builder that populates latency / token / correctness fields."""

    truncated = traffic_text[:max_traffic_len] if len(traffic_text) > max_traffic_len else traffic_text

    latency = LatencyBreakdown(
        alpha_ms=alpha_verdict.latency_ms,
        beta_ms=beta_verdict.latency_ms,
        consensus_ms=consensus_result.latency_ms,
        total_ms=alpha_verdict.latency_ms + beta_verdict.latency_ms + consensus_result.latency_ms,
    )

    token_usage = TokenUsage(
        alpha_tokens=alpha_verdict.token_count,
        beta_tokens=beta_verdict.token_count,
    )

    # Determine correctness if ground truth available
    # "escalate" verdicts are not counted as correct/incorrect (left as None)
    detection_correct: Optional[bool] = None
    if ground_truth_label is not None and consensus_result.final_verdict != "escalate":
        gt_is_attack = ground_truth_label.lower() not in ("normal", "benign")
        pred_is_attack = consensus_result.final_verdict == "attack"
        detection_correct = gt_is_attack == pred_is_attack

    return AuditRecord(
        timestamp=time.time(),
        sample_index=sample_index,
        traffic_text=truncated,
        alpha_verdict=alpha_verdict,
        beta_verdict=beta_verdict,
        consensus_result=consensus_result,
        ground_truth_label=ground_truth_label,
        detection_correct=detection_correct,
        latency=latency,
        token_usage=token_usage,
    )


# ---------------------------------------------------------------------------
# AuditLogger — streaming JSONL writer
# ---------------------------------------------------------------------------

class AuditLogger:
    """Streams AuditRecords to a JSONL file and keeps an in-memory buffer
    for aggregation queries."""

    def __init__(self, output_path: str | Path, buffer_in_memory: bool = True):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a", encoding="utf-8")
        self._buffer: List[AuditRecord] = [] if buffer_in_memory else []
        self._buffer_in_memory = buffer_in_memory
        self._count = 0

    # --- writing ---

    def log(self, record: AuditRecord) -> None:
        self._file.write(record.to_json() + "\n")
        self._file.flush()
        if self._buffer_in_memory:
            self._buffer.append(record)
        self._count += 1

    def log_batch(self, records: List[AuditRecord]) -> None:
        for r in records:
            self.log(r)

    # --- reading ---

    @property
    def records(self) -> List[AuditRecord]:
        """In-memory records (only available if buffer_in_memory=True)."""
        return self._buffer

    @property
    def count(self) -> int:
        return self._count

    def load_from_file(self) -> List[AuditRecord]:
        """Read all records from the JSONL file (for post-hoc analysis)."""
        records: List[AuditRecord] = []
        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(AuditRecord.from_dict(json.loads(line)))
        return records

    # --- aggregation ---

    def summary_by_consensus_type(self) -> Dict[str, Dict]:
        """Aggregate accuracy / FPR / count grouped by consensus_type."""
        from collections import defaultdict

        groups: Dict[str, List[AuditRecord]] = defaultdict(list)
        for r in self._buffer:
            groups[r.consensus_result.consensus_type].append(r)

        summary: Dict[str, Dict] = {}
        for ctype, recs in sorted(groups.items()):
            total = len(recs)
            has_gt = [r for r in recs if r.detection_correct is not None]
            correct = sum(1 for r in has_gt if r.detection_correct)
            # FP: predicted attack (not escalate) but actually benign
            fp = sum(
                1
                for r in recs
                if r.consensus_result.final_verdict == "attack"
                and r.ground_truth_label is not None
                and r.ground_truth_label.lower() in ("normal", "benign")
            )
            actual_benign = sum(
                1
                for r in recs
                if r.ground_truth_label is not None
                and r.ground_truth_label.lower() in ("normal", "benign")
            )
            summary[ctype] = {
                "count": total,
                "accuracy": correct / len(has_gt) if has_gt else None,
                "fpr": fp / actual_benign if actual_benign > 0 else None,
            }
        return summary

    def disagreement_samples(self) -> List[AuditRecord]:
        """Return records where agents disagreed (for RQ3 analysis)."""
        return [
            r
            for r in self._buffer
            if r.consensus_result.is_disagreement
        ]

    def efficiency_stats(self) -> Dict[str, float]:
        """Average latency and token usage across all buffered records."""
        if not self._buffer:
            return {}
        n = len(self._buffer)
        return {
            "avg_total_latency_ms": sum(r.latency.total_ms for r in self._buffer) / n,
            "avg_alpha_latency_ms": sum(r.latency.alpha_ms for r in self._buffer) / n,
            "avg_beta_latency_ms": sum(r.latency.beta_ms for r in self._buffer) / n,
            "avg_consensus_latency_ms": sum(r.latency.consensus_ms for r in self._buffer) / n,
            "avg_alpha_tokens": sum(r.token_usage.alpha_tokens for r in self._buffer) / n,
            "avg_beta_tokens": sum(r.token_usage.beta_tokens for r in self._buffer) / n,
            "avg_total_tokens": sum(r.token_usage.total_tokens for r in self._buffer) / n,
        }

    # --- lifecycle ---

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
